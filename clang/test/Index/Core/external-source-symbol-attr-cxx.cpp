// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

#define GEN_DECL_USR(mod_name, usr) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name, USR=usr, generated_declaration)))

class GEN_DECL_USR("Module", "s:Class") Class {
public:
    void method() GEN_DECL_USR("Module", "s:Class_method");
    void method2() GEN_DECL_USR("Module", "");

    static void staticMethod() GEN_DECL_USR("Module", "s:Class_staticMethod");
};

template<class T>
class GEN_DECL_USR("Module", "s:TemplateClass") TemplateClass {
public:
    void method() GEN_DECL_USR("Module", "s:TemplateClass_method");
};

void test() {
  Class c = Class();
  // CHECK: [[@LINE-1]]:3 | class/Swift | Class | s:Class |
  // CHECK: [[@LINE-2]]:13 | class/Swift | Class | s:Class |
  c.method();
  // CHECK: [[@LINE-1]]:5 | instance-method/Swift | method | s:Class_method |
  c.method2();
  // CHECK: [[@LINE-1]]:5 | instance-method/Swift | method2 | c:@M@Module@S@Class@F@method2# |
  Class::staticMethod();
  // CHECK: [[@LINE-1]]:10 | static-method/Swift | staticMethod | s:Class_staticMethod |
  // CHECK: [[@LINE-2]]:3 | class/Swift | Class | s:Class |
  TemplateClass<int> c2 = TemplateClass<int>();
  // CHECK: [[@LINE-1]]:3 | class(Gen)/Swift | TemplateClass | s:TemplateClass |
  // CHECK: [[@LINE-2]]:27 | class(Gen)/Swift | TemplateClass | s:TemplateClass |
  c2.method();
  // CHECK: [[@LINE-1]]:6 | instance-method/Swift | method | s:TemplateClass_method |
}
