// RUN: %check_clang_tidy %s performance-noexcept-move-constructor %t -- -- -fexceptions

struct C_1 {
 ~C_1() {}
 C_1(int a) {}
 C_1(C_1&& a) :C_1(5) {}
 // CHECK-FIXES: C_1(C_1&& a)  noexcept :C_1(5) {}
 C_1& operator=(C_1&&) { return *this; }
 // CHECK-FIXES: C_1& operator=(C_1&&)  noexcept { return *this; }
};

struct C_2 {
 ~C_2() {}
 C_2(C_2&& a);
// CHECK-FIXES: C_2(C_2&& a) noexcept ;
 C_2& operator=(C_2&&);
// CHECK-FIXES: C_2& operator=(C_2&&) noexcept ;
};

C_2::C_2(C_2&& a) {}
// CHECK-FIXES: C_2::C_2(C_2&& a)  noexcept {}
C_2& C_2::operator=(C_2&&) { return *this; }
// CHECK-FIXES: C_2& C_2::operator=(C_2&&)  noexcept { return *this; }

struct C_3 {
 ~C_3() {}
 C_3(C_3&& a);
// CHECK-FIXES: C_3(C_3&& a) noexcept ;
 C_3& operator=(C_3&& a);
// CHECK-FIXES: C_3& operator=(C_3&& a) noexcept ;
};

C_3::C_3(C_3&& a) = default;
C_3& C_3::operator=(C_3&& a) = default;

template <class T>
struct C_4 {
 C_4(C_4<T>&&) {}
// CHECK-FIXES: C_4(C_4<T>&&)  noexcept {}
 ~C_4() {}
 C_4& operator=(C_4&& a) = default;
};

template <class T>
struct C_5 {
 C_5(C_5<T>&&) {}
// CHECK-FIXES: C_5(C_5<T>&&)  noexcept {}
 ~C_5() {}
 auto operator=(C_5&& a)->C_5<T> = default;
};

template <class T>
struct C_6 {
 C_6(C_6<T>&&) {}
// CHECK-FIXES: C_6(C_6<T>&&)  noexcept {}
 ~C_6() {}
 auto operator=(C_6&& a)->C_6<T>;
// CHECK-FIXES: auto operator=(C_6&& a) noexcept ->C_6<T>;
};

template <class T>
auto C_6<T>::operator=(C_6<T>&& a) -> C_6<T> {}
// CHECK-FIXES: auto C_6<T>::operator=(C_6<T>&& a)  noexcept -> C_6<T> {}
