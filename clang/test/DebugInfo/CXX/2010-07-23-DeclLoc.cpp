// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// Require the template function declaration refer to the correct filename.
// First, locate the function decl in metadata, and pluck out the file handle:
// CHECK: !DISubprogram(name: "extract_dwarf_data_from_header
// CHECK-SAME:          file: [[FILE:![0-9]+]]
// Second: Require that filehandle refer to the correct filename:
// CHECK: [[FILE]] = !DIFile(filename: "decl_should_be_here.hpp"
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
namespace std {
  template<typename _Tp>   class auto_ptr   {
    _Tp* _M_ptr;
  public:
    typedef _Tp element_type;
    auto_ptr(element_type* __p = 0) throw() : _M_ptr(__p) {    }
    element_type&     operator*() const throw()     { return *_M_ptr; }
  };
}
class Pointer32 {
public:
  typedef uint32_t ptr_t;
  typedef uint32_t size_t;
};
class Pointer64 {
public:
  typedef uint64_t ptr_t;
  typedef uint64_t size_t;
};
class BigEndian {};
class LittleEndian {};
template <typename _SIZE, typename _ENDIANNESS> class SizeAndEndianness {
public:
  typedef _SIZE SIZE;
};
typedef SizeAndEndianness<Pointer32, LittleEndian> ISA32Little;
typedef SizeAndEndianness<Pointer32, BigEndian> ISA32Big;
typedef SizeAndEndianness<Pointer64, LittleEndian> ISA64Little;
typedef SizeAndEndianness<Pointer64, BigEndian> ISA64Big;
template <typename SIZE> class TRange {
protected:
  typename SIZE::ptr_t _location;
  typename SIZE::size_t _length;
  TRange(typename SIZE::ptr_t location, typename SIZE::size_t length) : _location(location), _length(length) {  }
};
template <typename SIZE, typename T> class TRangeValue : public TRange<SIZE> {
  T _value;
public:
  TRangeValue(typename SIZE::ptr_t location, typename SIZE::size_t length, T value) : TRange<SIZE>(location, length), _value(value) {};
};
template <typename SIZE> class TAddressRelocator {};
class CSCppSymbolOwner{};
class CSCppSymbolOwnerData{};
template <typename SIZE> class TRawSymbolOwnerData
{
  TRangeValue< SIZE, uint8_t* > _TEXT_text_section;
  const char* _dsym_path;
  uint32_t _dylib_current_version;
  uint32_t _dylib_compatibility_version;
public:
  TRawSymbolOwnerData() :
    _TEXT_text_section(0, 0, __null), _dsym_path(__null), _dylib_current_version(0), _dylib_compatibility_version(0) {}
};
template <typename SIZE_AND_ENDIANNESS> class TExtendedMachOHeader {};
# 16 "decl_should_be_here.hpp"
template <typename SIZE_AND_ENDIANNESS> void extract_dwarf_data_from_header(TExtendedMachOHeader<SIZE_AND_ENDIANNESS>& header,
                                                                            TRawSymbolOwnerData<typename SIZE_AND_ENDIANNESS::SIZE>& symbol_owner_data,
                                                                            TAddressRelocator<typename SIZE_AND_ENDIANNESS::SIZE>* address_relocator) {}
struct CSCppSymbolOwnerHashFunctor {
  void operator()(const CSCppSymbolOwner& symbol_owner) const {
# 97 "wrong_place_for_decl.cpp"
  }
};
template <typename SIZE_AND_ENDIANNESS> void create_symbol_owner_data_arch_specific(CSCppSymbolOwner* symbol_owner, const char* dsym_path) {
  typedef typename SIZE_AND_ENDIANNESS::SIZE SIZE;
  std::auto_ptr< TRawSymbolOwnerData<SIZE> > data(new TRawSymbolOwnerData<SIZE>());
  std::auto_ptr< TExtendedMachOHeader<SIZE_AND_ENDIANNESS> > header;
  extract_dwarf_data_from_header(*header, *data, (TAddressRelocator<typename SIZE_AND_ENDIANNESS::SIZE>*)__null);
}
void create_symbol_owner_data2(CSCppSymbolOwner* symbol_owner, const char* dsym_path) {
  create_symbol_owner_data_arch_specific< ISA32Little >(symbol_owner, dsym_path);
  create_symbol_owner_data_arch_specific< ISA32Big >(symbol_owner, dsym_path);
  create_symbol_owner_data_arch_specific< ISA64Little >(symbol_owner, dsym_path);
  create_symbol_owner_data_arch_specific< ISA64Big >(symbol_owner, dsym_path);
}
