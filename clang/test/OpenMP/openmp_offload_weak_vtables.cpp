// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -emit-llvm -disable-free -clear-ast-before-backend -main-file-name bigscience.i -mrelocation-model static -mframe-pointer=all -fmath-errno -ffp-contract=on -fno-rounding-math -mconstructor-aliases -funwind-tables=2 -target-cpu x86-64 -tune-cpu generic -debug-info-kind=constructor -dwarf-version=5 -debugger-tuning=gdb -Wno-openmp-mapping -fdeprecated-macro -ferror-limit 19 -fmessage-length=190 -fopenmp --offload-new-driver -fgnuc-version=4.2.1 -fskip-odr-check-in-gmf -fcxx-exceptions -fexceptions -fcolor-diagnostics --offload-targets=amdgcn-amd-amdhsa -faddrsig -fdwarf2-cfi-asm -o - -x c++-cpp-output %s | FileCheck %s
// CHECK: weak_odr

class Science
{

public:
   double init_value;

   virtual void compute(double *x, int N) { };
};

class HotScience : public Science
{

public:
    void compute(double *x, int N);

};

#pragma omp requires unified_shared_memory

int main(int argc, char *argv[]){

   HotScience myscienceclass;

   int N=10000;
   double *x = new double[N];

#pragma omp target teams loop
   for (int k = 0; k < N; k++){
      myscienceclass.compute(&x[k], N);
   }

   delete[] x;
}
