// RUN: %clang_cc1 -triple i686-windows-msvc -emit-llvm -std=c++14 \
// RUN:    -fno-threadsafe-statics -fms-extensions -O1 -mconstructor-aliases \
// RUN:    -disable-llvm-passes -o - %s -w -fms-compatibility-version=19.00 | \
// RUN:    FileCheck %s

struct CtorWithClosure {
  __declspec(dllexport) CtorWithClosure(...) {}
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FCtorWithClosure@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// CHECK:   %[[this_addr:.*]] = alloca %struct.CtorWithClosure*, align 4
// CHECK:   store %struct.CtorWithClosure* %this, %struct.CtorWithClosure** %[[this_addr]], align 4
// CHECK:   %[[this:.*]] = load %struct.CtorWithClosure*, %struct.CtorWithClosure** %[[this_addr]]
// CHECK:   call %struct.CtorWithClosure* (%struct.CtorWithClosure*, ...) @"??0CtorWithClosure@@QAA@ZZ"(%struct.CtorWithClosure* {{[^,]*}} %[[this]])
// CHECK:   ret void
};

struct CtorWithClosureOutOfLine {
  __declspec(dllexport) CtorWithClosureOutOfLine(...);
};
CtorWithClosureOutOfLine::CtorWithClosureOutOfLine(...) {}
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FCtorWithClosureOutOfLine@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat

#define DELETE_IMPLICIT_MEMBERS(ClassName) \
    ClassName(ClassName &&) = delete; \
    ClassName(ClassName &) = delete; \
    ~ClassName() = delete; \
    ClassName &operator=(ClassName &) = delete

struct __declspec(dllexport) ClassWithClosure {
  DELETE_IMPLICIT_MEMBERS(ClassWithClosure);
  ClassWithClosure(...) {}
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FClassWithClosure@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// CHECK:   %[[this_addr:.*]] = alloca %struct.ClassWithClosure*, align 4
// CHECK:   store %struct.ClassWithClosure* %this, %struct.ClassWithClosure** %[[this_addr]], align 4
// CHECK:   %[[this:.*]] = load %struct.ClassWithClosure*, %struct.ClassWithClosure** %[[this_addr]]
// CHECK:   call %struct.ClassWithClosure* (%struct.ClassWithClosure*, ...) @"??0ClassWithClosure@@QAA@ZZ"(%struct.ClassWithClosure* {{[^,]*}} %[[this]])
// CHECK:   ret void
};

template <typename T> struct TemplateWithClosure {
  TemplateWithClosure(int x = sizeof(T)) {}
};
extern template struct TemplateWithClosure<char>;
template struct __declspec(dllexport) TemplateWithClosure<char>;
extern template struct TemplateWithClosure<int>;
template struct __declspec(dllexport) TemplateWithClosure<int>;

// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_F?$TemplateWithClosure@D@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// CHECK:   call {{.*}} @"??0?$TemplateWithClosure@D@@QAE@H@Z"({{.*}}, i32 1)

// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_F?$TemplateWithClosure@H@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// CHECK:   call {{.*}} @"??0?$TemplateWithClosure@H@@QAE@H@Z"({{.*}}, i32 4)

template <typename T> struct __declspec(dllexport) ExportedTemplateWithClosure {
  ExportedTemplateWithClosure(int x = sizeof(T)) {}
};
template <> ExportedTemplateWithClosure<int>::ExportedTemplateWithClosure(int); // Don't try to emit the closure for a declaration.
template <> ExportedTemplateWithClosure<int>::ExportedTemplateWithClosure(int) {};
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_F?$ExportedTemplateWithClosure@H@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// CHECK:   call {{.*}} @"??0?$ExportedTemplateWithClosure@H@@QAE@H@Z"({{.*}}, i32 4)

struct __declspec(dllexport) NestedOuter {
  DELETE_IMPLICIT_MEMBERS(NestedOuter);
  NestedOuter(void *p = 0) {}
  struct __declspec(dllexport) NestedInner {
    DELETE_IMPLICIT_MEMBERS(NestedInner);
    NestedInner(void *p = 0) {}
  };
};

// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FNestedOuter@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FNestedInner@NestedOuter@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat

struct HasDtor {
  ~HasDtor();
  int o;
};
struct HasImplicitDtor1 { HasDtor o; };
struct HasImplicitDtor2 { HasDtor o; };
struct __declspec(dllexport) CtorClosureInline {
  CtorClosureInline(const HasImplicitDtor1 &v = {}) {}
};
struct __declspec(dllexport) CtorClosureOutOfLine {
  CtorClosureOutOfLine(const HasImplicitDtor2 &v = {});
};
CtorClosureOutOfLine::CtorClosureOutOfLine(const HasImplicitDtor2 &v) {}

// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FCtorClosureInline@@QAEXXZ"
// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"??1HasImplicitDtor1@@QAE@XZ"
// CHECK-LABEL: define weak_odr dso_local dllexport x86_thiscallcc void @"??_FCtorClosureOutOfLine@@QAEXXZ"
// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"??1HasImplicitDtor2@@QAE@XZ"
