// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv32 -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64 -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64-amd-amdhsa -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv32 -aux-triple x86_64-unknown-linux-gnu -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv32 -aux-triple x86_64-pc-windows-msvc -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64 -aux-triple x86_64-pc-windows-msvc -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -verify

// expected-no-diagnostics

namespace std
{
    enum class align_val_t : __SIZE_TYPE__ {};
    struct nothrow_t { explicit nothrow_t() = default; };
    extern nothrow_t const nothrow;
}

void* __attribute__((cdecl)) operator new(__SIZE_TYPE__);
void* __attribute__((cdecl)) operator new[](__SIZE_TYPE__);
void* __attribute__((cdecl)) operator new(__SIZE_TYPE__, ::std::align_val_t);
void* __attribute__((cdecl)) operator new[](__SIZE_TYPE__, ::std::align_val_t);

void __attribute__((cdecl)) operator delete(void*) noexcept;
void __attribute__((cdecl)) operator delete[](void*) noexcept;
void __attribute__((cdecl)) operator delete(void*, __SIZE_TYPE__) noexcept;
void __attribute__((cdecl)) operator delete[](void*, __SIZE_TYPE__) noexcept;
void __attribute__((cdecl)) operator delete(void*, ::std::align_val_t) noexcept;
void __attribute__((cdecl)) operator delete[](void*, ::std::align_val_t) noexcept;
void __attribute__((cdecl)) operator delete(void*, __SIZE_TYPE__, ::std::align_val_t) noexcept;
void __attribute__((cdecl)) operator delete[](void*, __SIZE_TYPE__, ::std::align_val_t) noexcept;
