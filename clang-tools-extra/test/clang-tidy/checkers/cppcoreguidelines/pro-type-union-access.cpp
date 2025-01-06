// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-union-access %t

union U {
  bool union_member1;
  char union_member2;
} u;

union W {
  template <class TP> operator TP *() const;
};

struct S {
  int non_union_member;
  union {
    bool union_member;
  };
  union {
    char union_member2;
  } u;
} s;


void f(char);
void f2(U);
void f3(U&);
void f4(U*);
W f5();

void check()
{
  u.union_member1 = true;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not access members of unions; consider using (boost::)variant instead [cppcoreguidelines-pro-type-union-access]
  auto b = u.union_member2;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not access members of unions; consider using (boost::)variant instead
  auto a = &s.union_member;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: do not access members of unions; consider using (boost::)variant instead
  f(s.u.union_member2);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not access members of unions; consider using (boost::)variant instead

  s.non_union_member = 2; // OK

  U u2 = u; // OK
  f2(u); // OK
  f3(u); // OK
  f4(&u); // OK
  void *ret = f5();
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: do not access members of unions; consider using (boost::)variant instead
}
