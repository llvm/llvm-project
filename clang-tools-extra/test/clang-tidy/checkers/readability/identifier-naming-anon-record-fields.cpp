// RUN: %check_clang_tidy -std=c++20 %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.CheckAnonFieldInParent: true, \
// RUN:     readability-identifier-naming.ClassConstantCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassConstantPrefix: 'k', \
// RUN:     readability-identifier-naming.ClassMemberCase: CamelCase, \
// RUN:     readability-identifier-naming.ConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.ConstantSuffix: '_CST', \
// RUN:     readability-identifier-naming.ConstexprVariableCase: lower_case, \
// RUN:     readability-identifier-naming.GlobalConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.GlobalVariableCase: lower_case, \
// RUN:     readability-identifier-naming.GlobalVariablePrefix: 'g_', \
// RUN:     readability-identifier-naming.LocalConstantCase: CamelCase, \
// RUN:     readability-identifier-naming.LocalConstantPrefix: 'k', \
// RUN:     readability-identifier-naming.LocalVariableCase: lower_case, \
// RUN:     readability-identifier-naming.MemberCase: CamelCase, \
// RUN:     readability-identifier-naming.MemberPrefix: 'm_', \
// RUN:     readability-identifier-naming.ConstantMemberCase: lower_case, \
// RUN:     readability-identifier-naming.PrivateMemberPrefix: '__', \
// RUN:     readability-identifier-naming.ProtectedMemberPrefix: '_', \
// RUN:     readability-identifier-naming.PublicMemberCase: lower_case, \
// RUN:     readability-identifier-naming.StaticConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.StaticVariableCase: camelBack, \
// RUN:     readability-identifier-naming.StaticVariablePrefix: 's_', \
// RUN:     readability-identifier-naming.VariableCase: lower_case, \
// RUN:     readability-identifier-naming.GlobalPointerCase: CamelCase, \
// RUN:     readability-identifier-naming.GlobalPointerSuffix: '_Ptr', \
// RUN:     readability-identifier-naming.GlobalConstantPointerCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.GlobalConstantPointerSuffix: '_Ptr', \
// RUN:     readability-identifier-naming.LocalPointerCase: CamelCase, \
// RUN:     readability-identifier-naming.LocalPointerPrefix: 'l_', \
// RUN:     readability-identifier-naming.LocalConstantPointerCase: CamelCase, \
// RUN:     readability-identifier-naming.LocalConstantPointerPrefix: 'lc_', \
// RUN:   }}'

static union {
  int global;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'global'
// CHECK-FIXES: {{^}}  int g_global;{{$}}

  const int global_const;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global constant 'global_const'
// CHECK-FIXES: {{^}}  const int GLOBAL_CONST;{{$}}

  int *global_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'global_ptr'
// CHECK-FIXES: {{^}}  int *GlobalPtr_Ptr;{{$}}

  int *const global_const_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global constant pointer 'global_const_ptr'
// CHECK-FIXES: {{^}}  int *const GLOBAL_CONST_PTR_Ptr;{{$}}
};

namespace ns {

static union {
  int ns_global;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ns_global'
// CHECK-FIXES: {{^}}  int g_ns_global;{{$}}

  const int ns_global_const;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global constant 'ns_global_const'
// CHECK-FIXES: {{^}}  const int NS_GLOBAL_CONST;{{$}}

  int *ns_global_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'ns_global_ptr'
// CHECK-FIXES: {{^}}  int *NsGlobalPtr_Ptr;{{$}}

  int *const ns_global_const_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global constant pointer 'ns_global_const_ptr'
// CHECK-FIXES: {{^}}  int *const NS_GLOBAL_CONST_PTR_Ptr;{{$}}
};

namespace {

union {
  int anon_ns_global;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'anon_ns_global'
// CHECK-FIXES: {{^}}  int g_anon_ns_global;{{$}}

  const int anon_ns_global_const;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global constant 'anon_ns_global_const'
// CHECK-FIXES: {{^}}  const int ANON_NS_GLOBAL_CONST;{{$}}

  int *anon_ns_global_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'anon_ns_global_ptr'
// CHECK-FIXES: {{^}}  int *AnonNsGlobalPtr_Ptr;{{$}}

  int *const anon_ns_global_const_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global constant pointer 'anon_ns_global_const_ptr'
// CHECK-FIXES: {{^}}  int *const ANON_NS_GLOBAL_CONST_PTR_Ptr;{{$}}
};

}

}


class Foo {
public:
  union {
    int PubMember;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for public member 'PubMember'
// CHECK-FIXES: {{^}}    int pub_member;{{$}}

    const int PubConstMember;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for constant member 'PubConstMember'
// CHECK-FIXES: {{^}}    const int pub_const_member;{{$}}

    int *PubPtrMember;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for public member 'PubPtrMember'
// CHECK-FIXES: {{^}}    int *pub_ptr_member;{{$}}

    int *const PubConstPtrMember;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for constant member 'PubConstPtrMember'
// CHECK-FIXES: {{^}}    int *const pub_const_ptr_member;{{$}}
  };

protected:
  union {
    int prot_member;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for protected member 'prot_member'
// CHECK-FIXES: {{^}}    int _prot_member;{{$}}

    const int prot_const_member;

    int *prot_ptr_member;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for protected member 'prot_ptr_member'
// CHECK-FIXES: {{^}}    int *_prot_ptr_member;{{$}}

    int *const prot_const_ptr_member;
  };


private:
  union {
    int pri_member;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for private member 'pri_member'
// CHECK-FIXES: {{^}}    int __pri_member;{{$}}

    const int pri_const_member;

    int *pri_ptr_member;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for private member 'pri_ptr_member'
// CHECK-FIXES: {{^}}    int *__pri_ptr_member;{{$}}

    int *const pri_const_ptr_member;
  };
};

void test() {
  union {
    int local;

    const int local_const;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for local constant 'local_const'
// CHECK-FIXES: {{^}}    const int kLocalConst;{{$}}

    int *local_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for local pointer 'local_ptr'
// CHECK-FIXES: {{^}}    int *l_LocalPtr;{{$}}

    int *const local_const_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for local constant pointer 'local_const_ptr'
// CHECK-FIXES: {{^}}    int *const lc_LocalConstPtr;{{$}}
  };

  static union {
    int local_static;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for static variable 'local_static'
// CHECK-FIXES: {{^}}    int s_localStatic;{{$}}

    const int local_static_const;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for static constant 'local_static_const'
// CHECK-FIXES: {{^}}    const int LOCAL_STATIC_CONST;{{$}}

    int *local_static_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for static variable 'local_static_ptr'
// CHECK-FIXES: {{^}}    int *s_localStaticPtr;{{$}}

    int *const local_static_const_ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for static constant 'local_static_const_ptr'
// CHECK-FIXES: {{^}}    int *const LOCAL_STATIC_CONST_PTR;{{$}}
  };
}
