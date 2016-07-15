//===- Librarian.cpp ------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for the Librarian.  The librarian creates and
// manages libraries of the Common Object File Format (COFF) object files.  It
// primarily is used for creating static libraries and import libraries.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Path.h"

#include <vector>

using namespace lld::coff;
using namespace llvm::COFF;
using namespace llvm::object;
using namespace llvm;

static bool is32bit() {
  switch (Config->Machine) {
  default:
    llvm_unreachable("unsupported machine");
  case IMAGE_FILE_MACHINE_AMD64:
    return false;
  case IMAGE_FILE_MACHINE_ARMNT:
  case IMAGE_FILE_MACHINE_I386:
    return true;
  }
}

static uint16_t getImgRelRelocation() {
  switch (Config->Machine) {
  default:
    llvm_unreachable("unsupported machine");
  case IMAGE_FILE_MACHINE_AMD64:
    return IMAGE_REL_AMD64_ADDR32NB;
  case IMAGE_FILE_MACHINE_ARMNT:
    return IMAGE_REL_ARM_ADDR32NB;
  case IMAGE_FILE_MACHINE_I386:
    return IMAGE_REL_I386_DIR32NB;
  }
}

template <class T> void append(std::vector<uint8_t> &B, const T &Data) {
  size_t S = B.size();
  B.resize(S + sizeof(T));
  memcpy(&B[S], &Data, sizeof(T));
}

static void writeStringTable(std::vector<uint8_t> &B,
                             ArrayRef<const std::string> Strings) {
  // The COFF string table consists of a 4-byte value which is the size of the
  // table, including the length field itself.  This value is followed by the
  // string content itself, which is an array of null-terminated C-style
  // strings.  The termination is important as they are referenced to by offset
  // by the symbol entity in the file format.

  std::vector<uint8_t>::size_type Pos = B.size();
  std::vector<uint8_t>::size_type Offset = B.size();

  // Skip over the length field, we will fill it in later as we will have
  // computed the length while emitting the string content itself.
  Pos += sizeof(uint32_t);

  for (const auto &S : Strings) {
    B.resize(Pos + S.length() + 1);
    strcpy(reinterpret_cast<char *>(&B[Pos]), S.c_str());
    Pos += S.length() + 1;
  }

  // Backfill the length of the table now that it has been computed.
  support::ulittle32_t Length(B.size() - Offset);
  memcpy(&B[Offset], &Length, sizeof(Length));
}

static std::string getImplibPath() {
  if (!Config->Implib.empty())
    return Config->Implib;
  SmallString<128> Out = StringRef(Config->OutputFile);
  sys::path::replace_extension(Out, ".lib");
  return Out.str();
}

static ImportNameType getNameType(StringRef Sym, StringRef ExtName) {
  if (Sym != ExtName)
    return IMPORT_NAME_UNDECORATE;
  if (Config->Machine == I386 && Sym.startswith("_"))
    return IMPORT_NAME_NOPREFIX;
  return IMPORT_NAME;
}

static std::string replace(StringRef S, StringRef From, StringRef To) {
  size_t Pos = S.find(From);
  assert(Pos != StringRef::npos);
  return (Twine(S.substr(0, Pos)) + To + S.substr(Pos + From.size())).str();
}

static const std::string NullImportDescriptorSymbolName =
    "__NULL_IMPORT_DESCRIPTOR";

namespace {
// This class constructs various small object files necessary to support linking
// symbols imported from a DLL.  The contents are pretty strictly defined and
// nearly entirely static.  The details of the structures files are defined in
// WINNT.h and the PE/COFF specification.
class ObjectFactory {
  using u16 = support::ulittle16_t;
  using u32 = support::ulittle32_t;

  BumpPtrAllocator Alloc;
  StringRef DLLName;
  StringRef Library;
  std::string ImportDescriptorSymbolName;
  std::string NullThunkSymbolName;

public:
  ObjectFactory(StringRef S)
      : DLLName(S), Library(S.drop_back(4)),
        ImportDescriptorSymbolName(("__IMPORT_DESCRIPTOR_" + Library).str()),
        NullThunkSymbolName(("\x7f" + Library + "_NULL_THUNK_DATA").str()) {}

  // Creates an Import Descriptor.  This is a small object file which contains a
  // reference to the terminators and contains the library name (entry) for the
  // import name table.  It will force the linker to construct the necessary
  // structure to import symbols from the DLL.
  NewArchiveMember createImportDescriptor(std::vector<uint8_t> &Buffer);

  // Creates a NULL import descriptor.  This is a small object file whcih
  // contains a NULL import descriptor.  It is used to terminate the imports
  // from a specific DLL.
  NewArchiveMember createNullImportDescriptor(std::vector<uint8_t> &Buffer);

  // Create a NULL Thunk Entry.  This is a small object file which contains a
  // NULL Import Address Table entry and a NULL Import Lookup Table Entry.  It
  // is used to terminate the IAT and ILT.
  NewArchiveMember createNullThunk(std::vector<uint8_t> &Buffer);

  // Create a short import file which is described in PE/COFF spec 7. Import
  // Library Format.
  NewArchiveMember createShortImport(StringRef Sym, uint16_t Ordinal,
                                     ImportNameType NameType, bool isData);
};
}

NewArchiveMember
ObjectFactory::createImportDescriptor(std::vector<uint8_t> &Buffer) {
  static const uint32_t NumberOfSections = 2;
  static const uint32_t NumberOfSymbols = 7;
  static const uint32_t NumberOfRelocations = 3;

  // COFF Header
  coff_file_header Header{
      u16(Config->Machine), u16(NumberOfSections), u32(0),
      u32(sizeof(Header) + (NumberOfSections * sizeof(coff_section)) +
          // .idata$2
          sizeof(coff_import_directory_table_entry) +
          NumberOfRelocations * sizeof(coff_relocation) +
          // .idata$4
          (DLLName.size() + 1)),
      u32(NumberOfSymbols), u16(0),
      u16(is32bit() ? IMAGE_FILE_32BIT_MACHINE : 0),
  };
  append(Buffer, Header);

  // Section Header Table
  static const coff_section SectionTable[NumberOfSections] = {
      {{'.', 'i', 'd', 'a', 't', 'a', '$', '2'},
       u32(0),
       u32(0),
       u32(sizeof(coff_import_directory_table_entry)),
       u32(sizeof(coff_file_header) + NumberOfSections * sizeof(coff_section)),
       u32(sizeof(coff_file_header) + NumberOfSections * sizeof(coff_section) +
           sizeof(coff_import_directory_table_entry)),
       u32(0),
       u16(NumberOfRelocations),
       u16(0),
       u32(IMAGE_SCN_ALIGN_4BYTES | IMAGE_SCN_CNT_INITIALIZED_DATA |
           IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)},
      {{'.', 'i', 'd', 'a', 't', 'a', '$', '6'},
       u32(0),
       u32(0),
       u32(DLLName.size() + 1),
       u32(sizeof(coff_file_header) + NumberOfSections * sizeof(coff_section) +
           sizeof(coff_import_directory_table_entry) +
           NumberOfRelocations * sizeof(coff_relocation)),
       u32(0),
       u32(0),
       u16(0),
       u16(0),
       u32(IMAGE_SCN_ALIGN_2BYTES | IMAGE_SCN_CNT_INITIALIZED_DATA |
           IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)},
  };
  append(Buffer, SectionTable);

  // .idata$2
  static const coff_import_directory_table_entry ImportDescriptor{
      u32(0), u32(0), u32(0), u32(0), u32(0),
  };
  append(Buffer, ImportDescriptor);

  static const coff_relocation RelocationTable[NumberOfRelocations] = {
      {u32(offsetof(coff_import_directory_table_entry, NameRVA)), u32(2),
       u16(getImgRelRelocation())},
      {u32(offsetof(coff_import_directory_table_entry, ImportLookupTableRVA)),
       u32(3), u16(getImgRelRelocation())},
      {u32(offsetof(coff_import_directory_table_entry, ImportAddressTableRVA)),
       u32(4), u16(getImgRelRelocation())},
  };
  append(Buffer, RelocationTable);

  // .idata$6
  auto S = Buffer.size();
  Buffer.resize(S + DLLName.size() + 1);
  memcpy(&Buffer[S], DLLName.data(), DLLName.size());
  Buffer[S + DLLName.size()] = '\0';

  // Symbol Table
  coff_symbol16 SymbolTable[NumberOfSymbols] = {
      {{{0, 0, 0, 0, 0, 0, 0, 0}},
       u32(0),
       u16(1),
       u16(0),
       IMAGE_SYM_CLASS_EXTERNAL,
       0},
      {{{'.', 'i', 'd', 'a', 't', 'a', '$', '2'}},
       u32(0),
       u16(1),
       u16(0),
       IMAGE_SYM_CLASS_SECTION,
       0},
      {{{'.', 'i', 'd', 'a', 't', 'a', '$', '6'}},
       u32(0),
       u16(2),
       u16(0),
       IMAGE_SYM_CLASS_STATIC,
       0},
      {{{'.', 'i', 'd', 'a', 't', 'a', '$', '4'}},
       u32(0),
       u16(0),
       u16(0),
       IMAGE_SYM_CLASS_SECTION,
       0},
      {{{'.', 'i', 'd', 'a', 't', 'a', '$', '5'}},
       u32(0),
       u16(0),
       u16(0),
       IMAGE_SYM_CLASS_SECTION,
       0},
      {{{0, 0, 0, 0, 0, 0, 0, 0}},
       u32(0),
       u16(0),
       u16(0),
       IMAGE_SYM_CLASS_EXTERNAL,
       0},
      {{{0, 0, 0, 0, 0, 0, 0, 0}},
       u32(0),
       u16(0),
       u16(0),
       IMAGE_SYM_CLASS_EXTERNAL,
       0},
  };
  reinterpret_cast<StringTableOffset &>(SymbolTable[0].Name).Offset =
      sizeof(uint32_t);
  reinterpret_cast<StringTableOffset &>(SymbolTable[5].Name).Offset =
      sizeof(uint32_t) + ImportDescriptorSymbolName.length() + 1;
  reinterpret_cast<StringTableOffset &>(SymbolTable[6].Name).Offset =
      sizeof(uint32_t) + ImportDescriptorSymbolName.length() + 1 +
      NullImportDescriptorSymbolName.length() + 1;
  append(Buffer, SymbolTable);

  // String Table
  writeStringTable(Buffer,
                   {ImportDescriptorSymbolName, NullImportDescriptorSymbolName,
                    NullThunkSymbolName});

  StringRef F{reinterpret_cast<const char *>(Buffer.data()), Buffer.size()};
  return {MemoryBufferRef(F, DLLName)};
}

NewArchiveMember
ObjectFactory::createNullImportDescriptor(std::vector<uint8_t> &Buffer) {
  static const uint32_t NumberOfSections = 1;
  static const uint32_t NumberOfSymbols = 1;

  // COFF Header
  coff_file_header Header{
      u16(Config->Machine), u16(NumberOfSections), u32(0),
      u32(sizeof(Header) + (NumberOfSections * sizeof(coff_section)) +
          // .idata$3
          sizeof(coff_import_directory_table_entry)),
      u32(NumberOfSymbols), u16(0),
      u16(is32bit() ? IMAGE_FILE_32BIT_MACHINE : 0),
  };
  append(Buffer, Header);

  // Section Header Table
  static const coff_section SectionTable[NumberOfSections] = {
      {{'.', 'i', 'd', 'a', 't', 'a', '$', '3'},
       u32(0),
       u32(0),
       u32(sizeof(coff_import_directory_table_entry)),
       u32(sizeof(coff_file_header) +
           (NumberOfSections * sizeof(coff_section))),
       u32(0),
       u32(0),
       u16(0),
       u16(0),
       u32(IMAGE_SCN_ALIGN_4BYTES | IMAGE_SCN_CNT_INITIALIZED_DATA |
           IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)},
  };
  append(Buffer, SectionTable);

  // .idata$3
  static const coff_import_directory_table_entry ImportDescriptor{
      u32(0), u32(0), u32(0), u32(0), u32(0),
  };
  append(Buffer, ImportDescriptor);

  // Symbol Table
  coff_symbol16 SymbolTable[NumberOfSymbols] = {
      {{{0, 0, 0, 0, 0, 0, 0, 0}},
       u32(0),
       u16(1),
       u16(0),
       IMAGE_SYM_CLASS_EXTERNAL,
       0},
  };
  reinterpret_cast<StringTableOffset &>(SymbolTable[0].Name).Offset =
      sizeof(uint32_t);
  append(Buffer, SymbolTable);

  // String Table
  writeStringTable(Buffer, {NullImportDescriptorSymbolName});

  StringRef F{reinterpret_cast<const char *>(Buffer.data()), Buffer.size()};
  return {MemoryBufferRef(F, DLLName)};
}

NewArchiveMember ObjectFactory::createNullThunk(std::vector<uint8_t> &Buffer) {
  static const uint32_t NumberOfSections = 2;
  static const uint32_t NumberOfSymbols = 1;

  // COFF Header
  coff_file_header Header{
      u16(Config->Machine), u16(NumberOfSections), u32(0),
      u32(sizeof(Header) + (NumberOfSections * sizeof(coff_section)) +
          // .idata$5
          sizeof(export_address_table_entry) +
          // .idata$4
          sizeof(export_address_table_entry)),
      u32(NumberOfSymbols), u16(0),
      u16(is32bit() ? IMAGE_FILE_32BIT_MACHINE : 0),
  };
  append(Buffer, Header);

  // Section Header Table
  static const coff_section SectionTable[NumberOfSections] = {
      {{'.', 'i', 'd', 'a', 't', 'a', '$', '5'},
       u32(0),
       u32(0),
       u32(sizeof(export_address_table_entry)),
       u32(sizeof(coff_file_header) + NumberOfSections * sizeof(coff_section)),
       u32(0),
       u32(0),
       u16(0),
       u16(0),
       u32(IMAGE_SCN_ALIGN_4BYTES | IMAGE_SCN_CNT_INITIALIZED_DATA |
           IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)},
      {{'.', 'i', 'd', 'a', 't', 'a', '$', '4'},
       u32(0),
       u32(0),
       u32(sizeof(export_address_table_entry)),
       u32(sizeof(coff_file_header) + NumberOfSections * sizeof(coff_section) +
           sizeof(export_address_table_entry)),
       u32(0),
       u32(0),
       u16(0),
       u16(0),
       u32(IMAGE_SCN_ALIGN_4BYTES | IMAGE_SCN_CNT_INITIALIZED_DATA |
           IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)},
  };
  append(Buffer, SectionTable);

  // .idata$5
  static const export_address_table_entry ILT{u32(0)};
  append(Buffer, ILT);

  // .idata$4
  static const export_address_table_entry IAT{u32(0)};
  append(Buffer, IAT);

  // Symbol Table
  coff_symbol16 SymbolTable[NumberOfSymbols] = {
      {{{0, 0, 0, 0, 0, 0, 0, 0}},
       u32(0),
       u16(1),
       u16(0),
       IMAGE_SYM_CLASS_EXTERNAL,
       0},
  };
  reinterpret_cast<StringTableOffset &>(SymbolTable[0].Name).Offset =
      sizeof(uint32_t);
  append(Buffer, SymbolTable);

  // String Table
  writeStringTable(Buffer, {NullThunkSymbolName});

  StringRef F{reinterpret_cast<const char *>(Buffer.data()), Buffer.size()};
  return {MemoryBufferRef{F, DLLName}};
}

NewArchiveMember ObjectFactory::createShortImport(StringRef Sym,
                                                  uint16_t Ordinal,
                                                  ImportNameType NameType,
                                                  bool isData) {
  size_t ImpSize = DLLName.size() + Sym.size() + 2; // +2 for NULs
  size_t Size = sizeof(coff_import_header) + ImpSize;
  char *Buf = Alloc.Allocate<char>(Size);
  memset(Buf, 0, Size);
  char *P = Buf;

  // Write short import library.
  auto *Imp = reinterpret_cast<coff_import_header *>(P);
  P += sizeof(*Imp);
  Imp->Sig2 = 0xFFFF;
  Imp->Machine = Config->Machine;
  Imp->SizeOfData = ImpSize;
  if (Ordinal > 0)
    Imp->OrdinalHint = Ordinal;
  Imp->TypeInfo = (isData ? IMPORT_DATA : IMPORT_CODE);
  Imp->TypeInfo |= NameType << 2;

  // Write symbol name and DLL name.
  memcpy(P, Sym.data(), Sym.size());
  P += Sym.size() + 1;
  memcpy(P, DLLName.data(), DLLName.size());

  return {MemoryBufferRef(StringRef(Buf, Size), DLLName)};
}

// Creates an import library for a DLL. In this function, we first
// create an empty import library using lib.exe and then adds short
// import files to that file.
void lld::coff::writeImportLibrary() {
  std::vector<NewArchiveMember> Members;

  std::string Path = getImplibPath();
  std::string DLLName = llvm::sys::path::filename(Config->OutputFile);
  ObjectFactory OF(DLLName);

  std::vector<uint8_t> ImportDescriptor;
  Members.push_back(OF.createImportDescriptor(ImportDescriptor));

  std::vector<uint8_t> NullImportDescriptor;
  Members.push_back(OF.createNullImportDescriptor(NullImportDescriptor));

  std::vector<uint8_t> NullThunk;
  Members.push_back(OF.createNullThunk(NullThunk));

  for (Export &E : Config->Exports) {
    if (E.Private)
      continue;

    ImportNameType Type = getNameType(E.SymbolName, E.Name);
    std::string Name = E.ExtName.empty()
                           ? std::string(E.SymbolName)
                           : replace(E.SymbolName, E.Name, E.ExtName);
    Members.push_back(OF.createShortImport(Name, E.Ordinal, Type, E.Data));
  }

  std::pair<StringRef, std::error_code> Result =
      writeArchive(Path, Members, /*WriteSymtab*/ true, object::Archive::K_GNU,
                   /*Deterministic*/ true, /*Thin*/ false);
  if (auto EC = Result.second)
    fatal(EC, "failed to write " + Path);
}
