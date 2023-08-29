// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

@class CPDestUser;

CPDestUser* FUNC();

// CHECK: {{call.* @objc_assign_global}}
CPDestUser* globalUser = FUNC();

// CHECK: {{call.* @objc_assign_weak}}
__weak CPDestUser* weakUser = FUNC();


// CHECK: {{call.* @objc_assign_global}}
static CPDestUser* staticUser = FUNC();

CPDestUser* GetDestUser()
{
// CHECK: {{call.* @objc_assign_global}}
	static CPDestUser* gUser = FUNC();
// CHECK: {{call.* @objc_assign_weak}}
	static __weak CPDestUser* wUser = FUNC();
        if (wUser)
          return wUser;
        if (staticUser)
	  return staticUser;
	return gUser;
}
