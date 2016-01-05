// RUN: %clang_cc1 %s -triple %itanium_abi_triple -std=c++11 -emit-llvm -o - | FileCheck %s

// Test optnone on template instantiations.

//-- Effect of optnone on generic add template function.

template <typename T> T template_normal(T a)
{
  return a + a;
}

template <typename T> __attribute__((optnone)) T template_optnone(T a)
{
  return a + a + a;
}

// This function should cause instantiations of each template, one marked
// with the 'optnone' attribute.
int container(int i)
{
  return template_normal<int>(i) + template_optnone<int>(i);
}

// CHECK: @_Z15template_normalIiET_S0_({{.*}}) [[NORMAL:#[0-9]+]]
// CHECK: @_Z16template_optnoneIiET_S0_({{.*}}) [[OPTNONE:#[0-9]+]]


//-- Effect of optnone on a partial specialization.
//   FIRST TEST: a method becomes marked with optnone in the specialization.

template <typename T, typename U> class template_normal_base {
public:
  T method(T t, U u) 
  {
    return t + static_cast<T>(u);
  }
};

template <typename U> class template_normal_base<int, U>
{
public:
  __attribute__((optnone)) int method (int t, U u)
  {
    return t - static_cast<int>(u);
  }
};

// This function should cause an instantiation of the full template (whose
// method is not marked optnone) and an instantiation of the partially
// specialized template (whose method is marked optnone).
void container2() 
{
  int y = 2;
  float z = 3.0;
  template_normal_base<float, int> class_normal;
  template_normal_base<int, float> class_optnone;
  float r1 = class_normal.method(z, y);
  float r2 = class_optnone.method(y, z);
}

// CHECK: @_ZN20template_normal_baseIfiE6methodEfi({{.*}}) [[NORMAL]]
// CHECK: @_ZN20template_normal_baseIifE6methodEif({{.*}}) [[OPTNONE]]


//-- Effect of optnone on a partial specialization.
//   SECOND TEST: a method loses optnone in the specialization.

template <typename T, typename U> class template_optnone_base {
public:
  __attribute__((optnone)) T method(T t, U u) 
  {
    return t + static_cast<T>(u);
  }
};

template <typename U> class template_optnone_base<int, U>
{
public:
  int method (int t, U u)
  {
    return t - static_cast<int>(u);
  }
};

// This function should cause an instantiation of the full template (whose
// method is marked optnone) and an instantiation of the partially
// specialized template (whose method is not marked optnone).
void container3() 
{
  int y = 2;
  float z = 3.0;
  template_optnone_base<float, int> class_optnone;
  template_optnone_base<int, float> class_normal;
  float r1 = class_optnone.method(z, y);
  float r2 = class_normal.method(y, z);
}

// CHECK: @_ZN21template_optnone_baseIfiE6methodEfi({{.*}}) [[OPTNONE]]
// CHECK: @_ZN21template_optnone_baseIifE6methodEif({{.*}}) [[NORMAL]]


// CHECK: attributes [[NORMAL]] =
// CHECK-SAME-NOT: optnone
// CHECK: attributes [[OPTNONE]] = {{.*}} optnone
