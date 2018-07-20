// RUN: %clang_cc1 -emit-llvm -triple i686-pc-win32 -fms-extensions -o - %s | FileCheck %s

extern "C" {

#pragma const_seg(".my_const")
#pragma bss_seg(".my_bss")
int D = 1;
#pragma data_seg(".data")
int a = 1;
#pragma data_seg(push, label, ".data2")
extern const int b;
const int b = 1;
const char* s = "my string!";
#pragma data_seg(push, ".my_seg")
int c = 1;
#pragma data_seg(pop, label)
int d = 1;
int e;
#pragma bss_seg(".c")
int f;
void g(void){}
#pragma code_seg(".my_code")
void h(void){}
#pragma bss_seg()
int i;
#pragma bss_seg(".bss1")
#pragma bss_seg(push, test, ".bss2")
#pragma bss_seg()
#pragma bss_seg()
int TEST1;
#pragma bss_seg(pop)
int TEST2;


// Check "save-restore" of pragma stacks.
struct Outer {
  void f() {
    #pragma bss_seg(push, ".bss3")
    #pragma code_seg(push, ".my_code1")
    #pragma const_seg(push, ".my_const1")
    #pragma data_seg(push, ".data3")
    struct Inner {
      void g() {
        #pragma bss_seg(push, ".bss4")
        #pragma code_seg(push, ".my_code2")
        #pragma const_seg(push, ".my_const2")
        #pragma data_seg(push, ".data4")
      }
    };
  }
};

void h2(void) {} // should be in ".my_code"
int TEST3; // should be in ".bss1"
int d2 = 1; // should be in ".data"
extern const int b2; // should be in ".my_const"
const int b2 = 1;

#pragma section("read_flag_section", read)
// Even though they are not declared const, these become constant since they are
// in a read-only section.
__declspec(allocate("read_flag_section")) int unreferenced = 0;
extern __declspec(allocate("read_flag_section")) int referenced = 42;
int *user() { return &referenced; }

#pragma section("no_section_attributes")
// A pragma section with no section attributes is read/write.
__declspec(allocate("no_section_attributes")) int implicitly_read_write = 42;

#pragma section("long_section", long)
// Pragma section ignores "long".
__declspec(allocate("long_section")) long long_var = 42;

#pragma section("short_section", short)
// Pragma section ignores "short".
__declspec(allocate("short_section")) short short_var = 42;
}

//CHECK: @D = dso_local global i32 1
//CHECK: @a = dso_local global i32 1, section ".data"
//CHECK: @b = dso_local constant i32 1, section ".my_const"
//CHECK: @[[MYSTR:.*]] = {{.*}} unnamed_addr constant [11 x i8] c"my string!\00"
//CHECK: @s = dso_local global i8* getelementptr inbounds ([11 x i8], [11 x i8]* @[[MYSTR]], i32 0, i32 0), section ".data2"
//CHECK: @c = dso_local global i32 1, section ".my_seg"
//CHECK: @d = dso_local global i32 1, section ".data"
//CHECK: @e = dso_local global i32 0, section ".my_bss"
//CHECK: @f = dso_local global i32 0, section ".c"
//CHECK: @i = dso_local global i32 0
//CHECK: @TEST1 = dso_local global i32 0
//CHECK: @TEST2 = dso_local global i32 0, section ".bss1"
//CHECK: @TEST3 = dso_local global i32 0, section ".bss1"
//CHECK: @d2 = dso_local global i32 1, section ".data"
//CHECK: @b2 = dso_local constant i32 1, section ".my_const"
//CHECK: @unreferenced = dso_local constant i32 0, section "read_flag_section"
//CHECK: @referenced = dso_local constant i32 42, section "read_flag_section"
//CHECK: @implicitly_read_write = dso_local global i32 42, section "no_section_attributes"
//CHECK: @long_var = dso_local global i32 42, section "long_section"
//CHECK: @short_var = dso_local global i16 42, section "short_section"
//CHECK: define dso_local void @g()
//CHECK: define dso_local void @h() {{.*}} section ".my_code"
//CHECK: define dso_local void @h2() {{.*}} section ".my_code"
