///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s --check-prefix=CK1
//
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s --check-prefix=CK2
//
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s --check-prefix=CK3
//
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s --check-prefix=CK4
//
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52 -stdlib=libc++
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 -stdlib=libc++ | FileCheck %s --check-prefix=CK5
// expected-no-diagnostics

#ifndef HEADER
#define HEADER
#ifdef CK1

// Make sure both host and device compilation emit vtable for Dervied
// CK1-DAG: $_ZN7DerivedD1Ev = comdat any
// CK1-DAG: $_ZN7DerivedD0Ev = comdat any
// CK1-DAG: $_ZN7Derived5BaseAEi = comdat any
// CK1-DAG: $_ZN7Derived8DerivedBEv = comdat any
// CK1-DAG: $_ZN7DerivedD2Ev = comdat any
// CK1-DAG: $_ZN4BaseD2Ev = comdat any
// CK1-DAG: $_ZTV7Derived = comdat any
class Base {
public:
  virtual ~Base() = default;
  virtual void BaseA(int a) { }
};

// CK1: @_ZTV7Derived = linkonce_odr unnamed_addr constant { [6 x ptr] }
class Derived : public Base {
public:
  ~Derived() override = default;
  void BaseA(int a) override { x = a; }
  virtual void DerivedB() { }
private:
  int x;
};

int main() {

  Derived d;
  Base& c = d;
  int a = 50;
  // Should emit vtable for Derived since d is added to map clause
#pragma omp target data map (to: d, a)
  {
 #pragma omp target map(d)
     {
       c.BaseA(a);
     }
  }
  return 0;
}

#endif // CK1

#ifdef CK2

namespace {

// Make sure both host and device compilation emit vtable for Dervied
// CK2-DAG: @_ZTVN12_GLOBAL__N_17DerivedE
// CK2-DAG: @_ZN12_GLOBAL__N_17DerivedD1Ev
// CK2-DAG: @_ZN12_GLOBAL__N_17DerivedD0Ev
// CK2-DAG: @_ZN12_GLOBAL__N_17Derived5BaseAEi
// CK2-DAG: @_ZN12_GLOBAL__N_17Derived8DerivedBEv
class Base {
public:
  virtual ~Base() = default;
  virtual void BaseA(int a) { }
};

class Derived : public Base {
public:
  ~Derived() override = default;
  void BaseA(int a) override { x = a; }
  virtual void DerivedB() { }
private:
  int x;
};

};

int main() {

  Derived d;
  Base& c = d;
  int a = 50;
#pragma omp target data map (to: d, a)
  {
 #pragma omp target
     {
       c.BaseA(a);
     }
  }
  return 0;
}

#endif // CK2

#ifdef CK3

// CK3-DAG: @_ZTV6Base_1
// CK3-DAG: @_ZTV7Derived
// CK3-DAG: @_ZTV6Base_2
#pragma omp begin declare target

class Base_1 {
public:
  virtual void foo() { }
  virtual void bar() { }
};

class Base_2 {
public:
  virtual void foo() { }
  virtual void bar() { }
};

class Derived : public Base_1, public Base_2 {
public:
  virtual void foo() override { }
  virtual void bar() override { }
};

#pragma omp end declare target

int main() {
  Base_1 base;
  Derived derived;

  // Make sure we emit vtable for parent class (Base_1 and Base_2)
#pragma omp target data map(derived)
    {
      Base_1 *p1 = &derived;

#pragma omp target
      {
        p1->foo();
        p1->bar();
      }
    }
  return 0;
}

#endif // CK3
 
#ifdef CK4

// CK4-DAG: @_ZTV3Car
// CK4-DAG: @_ZTV6Engine
// CK4-DAG: @_ZTV6Wheels
// CK4-DAG: @_ZTV7Vehicle
// CK4-DAG: @_ZTV5Brand
class Engine {
public:
  Engine(const char *type) : type(type) {}
  virtual ~Engine() {}

  virtual void start() const { }

protected:
  const char *type;
};

class Wheels {
public:
  Wheels(int count) : count(count) {}
  virtual ~Wheels() {}

  virtual void roll() const { }

protected:
  int count;
};

class Vehicle {
public:
  Vehicle(int speed) : speed(speed) {}
  virtual ~Vehicle() {}

  virtual void move() const { }

protected:
  int speed;
};

class Brand {
public:
  Brand(const char *brandName) : brandName(brandName) {}
  virtual ~Brand() {}

  void showBrand() const { }

protected:
  const char *brandName;
};

class Car : public Vehicle, public Brand {
public:
  Car(const char *brand, int speed, const char *engineType, int wheelCount)
      : Vehicle(speed), Brand(brand), engine(engineType), wheels(wheelCount) {}

  void move() const override { }

  void drive() const {
    showBrand();
    engine.start();
    wheels.roll();
    move();
  }

private:
  Engine engine;
  Wheels wheels;
};

int main() {
  Car myActualCar("Ford", 100, "Hybrid", 4);

  // Make sure we emit VTable for dynamic class as field
#pragma omp target map(myActualCar)
  {
    myActualCar.drive();
  }
  return 0;
}

#endif // CK4

#ifdef CK5

// CK5-DAG: @_ZTV7Derived
// CK5-DAG: @_ZTV4Base
template <typename T>
class Container {
private:
T value;
public:
Container() : value() {}
Container(T val) : value(val) {}

T getValue() const { return value; }

void setValue(T val) { value = val; }
};

class Base {
public:
    virtual void foo() {}
};
class Derived : public Base {};

class Test {
public:
    Container<Derived> v;
};

int main() {
  Test test;
  Derived d;
  test.v.setValue(d);

// Make sure we emit VTable for type indirectly (template specialized type)
#pragma omp target map(test)
  {
      test.v.getValue().foo();
  }
  return 0;
}

#endif // CK5
#endif
