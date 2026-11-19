// REQUIRES: asan-dynamic-runtime
// UNSUPPORTED: target={{.*-windows-gnu}}
// NOTE: Do not autogenerate. These checks cover Windows version-resource fields.
// RUN: %clang_cl %s %Fe%t /link version.lib
// RUN: %run %t %asan_lib 2>&1 | FileCheck %s

#include <cstdio>
#include <string>
#include <vector>
#include <windows.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  std::string Path(argv[1]);
  Path.replace(Path.rfind('.'), std::string::npos, ".dll");

  DWORD Size = GetFileVersionInfoSizeA(Path.c_str(), nullptr);
  if (!Size)
    return 1;
  std::vector<char> Data(Size);
  if (!GetFileVersionInfoA(Path.c_str(), 0, Size, Data.data()))
    return 1;

  VS_FIXEDFILEINFO *Info;
  UINT InfoLength;
  if (!VerQueryValueA(Data.data(), "\\", reinterpret_cast<void **>(&Info),
                      &InfoLength) ||
      InfoLength < sizeof(VS_FIXEDFILEINFO) ||
      Info->dwSignature != VS_FFI_SIGNATURE ||
      !(Info->dwFileVersionMS | Info->dwFileVersionLS) ||
      Info->dwProductVersionMS != Info->dwFileVersionMS ||
      Info->dwProductVersionLS != Info->dwFileVersionLS)
    return 1;
  puts("numeric file/product versions match");

  struct Translation {
    WORD Language;
    WORD CodePage;
  };
  Translation *Translations;
  UINT Length;
  if (!VerQueryValueA(Data.data(), "\\VarFileInfo\\Translation",
                      reinterpret_cast<void **>(&Translations), &Length) ||
      Length < sizeof(Translation))
    return 1;

  auto ReadString = [&](const char *Field) {
    char Query[128];
    snprintf(Query, sizeof(Query), "\\StringFileInfo\\%04x%04x\\%s",
             Translations[0].Language, Translations[0].CodePage, Field);
    char *Value;
    if (!VerQueryValueA(Data.data(), Query, reinterpret_cast<void **>(&Value),
                        &Length))
      return std::string();
    return std::string(Value);
  };

  std::string Description = ReadString("FileDescription");
  std::string Copyright = ReadString("LegalCopyright");
  std::string Filename = ReadString("OriginalFilename");
  printf("description: %s\ncopyright: %s\nfilename: %s\n", Description.c_str(),
         Copyright.c_str(), Filename.c_str());
  return Filename != Path.substr(Path.find_last_of("/\\") + 1);
}

// CHECK: numeric file/product versions match
// CHECK: description: AddressSanitizer runtime
// CHECK: copyright: Copyright (c) LLVM contributors.
// CHECK: filename: clang_rt.asan_dynamic-{{.*}}.dll
