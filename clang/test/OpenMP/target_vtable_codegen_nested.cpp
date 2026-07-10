// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s
// expected-no-diagnostics

// CHECK-DAG: @_ZTV3Car
// CHECK-DAG: @_ZTV6Engine
// CHECK-DAG: @_ZTV6Wheels
// CHECK-DAG: @_ZTV7Vehicle
// CHECK-DAG: @_ZTV5Brand
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
