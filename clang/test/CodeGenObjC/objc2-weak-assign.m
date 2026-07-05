// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -e "objc_assign_weak" %t | grep -e "call" | count 6
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -e "objc_assign_weak" %t | grep -e "call" | count 6

__weak id* x;
id* __weak y;
id* __weak* z;

__weak id* a1[20];
id* __weak a2[30];
id** __weak a3[40];

void foo (__weak id *param) {
 *param = 0;
}

int main(void)
{
	*x = 0;
	*y = 0;
        **z = 0;

        a1[3] = 0;
        a2[3] = 0;
        a3[3][4] = 0;
}

