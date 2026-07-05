// RUN: %clang_cc1 -triple x86_64-darwin-apple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

int bar(void);
int foo(int i) {
	int j = 0;
	if (i) {
		j = bar();
	} 
	else
	{
          // CHECK: add nsw
          // CHECK-NEXT: store i32 %{{[a-zA-Z0-9]+}}
          // CHECK-NOT:  br label %{{[a-zA-Z0-9\.]+}}, !dbg 
		j = bar() + 2;
	}
	return j;
}
