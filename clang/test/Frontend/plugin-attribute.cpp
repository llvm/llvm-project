// RUN: %clang -fplugin=%llvmshlibdir/Attribute%pluginext -emit-llvm -S %s -o - 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
// RUN: not %clang -fplugin=%llvmshlibdir/Attribute%pluginext -emit-llvm -DBAD_ATTRIBUTE -S %s -o - 2>&1 | FileCheck %s --check-prefix=BADATTRIBUTE
// REQUIRES: plugins, examples

void fn1a() __attribute__((example)) { }
[[example]] void fn1b() { }
[[plugin::example]] void fn1c() { }
void fn2() __attribute__((example("somestring"))) { }
// ATTRIBUTE: warning: 'example' attribute only applies to functions
int var1 __attribute__((example("otherstring"))) = 1;

// ATTRIBUTE: [[STR1_VAR:@.+]] = private unnamed_addr constant [10 x i8] c"example()\00"
// ATTRIBUTE: [[STR2_VAR:@.+]] = private unnamed_addr constant [20 x i8] c"example(somestring)\00"
// ATTRIBUTE: @llvm.global.annotations = {{.*}}@{{.*}}fn1a{{.*}}[[STR1_VAR]]{{.*}}@{{.*}}fn1b{{.*}}[[STR1_VAR]]{{.*}}@{{.*}}fn1c{{.*}}[[STR1_VAR]]{{.*}}@{{.*}}fn2{{.*}}[[STR2_VAR]]

#ifdef BAD_ATTRIBUTE
class Example {
  // BADATTRIBUTE: error: 'example' attribute only allowed at file scope
  void __attribute__((example)) fn3();
};
// BADATTRIBUTE: error: 'example' attribute requires a string
void fn4() __attribute__((example(123))) { }
// BADATTRIBUTE: error: 'example' attribute takes no more than 1 argument
void fn5() __attribute__((example("a","b"))) { }
#endif
