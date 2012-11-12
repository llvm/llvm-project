// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10.0.0 -o - %s | FileCheck %s

// PR7050
template<class T> struct X0 : public T { };

template <class T>
struct X1
{
     static T & instance;
    // include this to provoke instantiation at pre-execution time
    static void use(T const &) {}
     static T & get() {
        static X0<T> t;
        use(instance);
        return static_cast<T &>(t);
    }
};

// CHECK: @_ZN2X1I2X2I1BEE8instanceE = weak_odr global %struct.X2* null, align 8
// CHECJ: @_ZN2X1I2X2I1AEE8instanceE = weak_odr global %struct.X2* null, align 8
template<class T> T & X1<T>::instance = X1<T>::get();

class A { };
class B : public A { };

template<typename T> struct X2 {};
X2< B > bg = X1< X2< B > >::get(); 
X2< A > ag = X1< X2< A > >::get();
