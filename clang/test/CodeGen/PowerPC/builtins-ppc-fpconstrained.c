// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-gnu-linux -target-feature +vsx \
// RUN: -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-UNCONSTRAINED %s
// RUN: %clang_cc1 -triple powerpc64le-gnu-linux -target-feature +vsx \
// RUN:  -ffp-exception-behavior=strict -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefix=CHECK-CONSTRAINED -vv %s
// RUN: %clang_cc1 -triple powerpc64le-gnu-linux -target-feature +vsx \
// RUN: -S -o - %s | \
// RUN: FileCheck --check-prefix=CHECK-ASM --check-prefix=NOT-FIXME-CHECK  %s
// RUN: %clang_cc1 -triple powerpc64le-gnu-linux -target-feature +vsx \
// RUN: -S -ffp-exception-behavior=strict \
// RUN: -o - %s | FileCheck --check-prefix=CHECK-ASM \
// RUN: --check-prefix=FIXME-CHECK  %s
// RUN: %clang_cc1 -triple powerpc -ffp-exception-behavior=strict \
// RUN: -target-feature +vsx -fexperimental-strict-floating-point -emit-llvm \
// RUN: %s -o - | FileCheck --check-prefix=CHECK-CONSTRAINED %s

typedef __attribute__((vector_size(4 * sizeof(float)))) float vec_float;
typedef __attribute__((vector_size(2 * sizeof(double)))) double vec_double;

volatile vec_double vd;
volatile vec_float vf;

void test_float(void) {
  vf = __builtin_vsx_xvsqrtsp(vf);
  // CHECK-LABEL: try-xvsqrtsp
  // CHECK-UNCONSTRAINED: @llvm.sqrt.v4f32(<4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.sqrt.v4f32(<4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvsqrtsp

  vd = __builtin_vsx_xvsqrtdp(vd);
  // CHECK-LABEL: try-xvsqrtdp
  // CHECK-UNCONSTRAINED: @llvm.sqrt.v2f64(<2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.sqrt.v2f64(<2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvsqrtdp

  vf = __builtin_vsx_xvrspim(vf);
  // CHECK-LABEL: try-xvrspim
  // CHECK-UNCONSTRAINED: @llvm.floor.v4f32(<4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.floor.v4f32(<4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrspim

  vd = __builtin_vsx_xvrdpim(vd);
  // CHECK-LABEL: try-xvrdpim
  // CHECK-UNCONSTRAINED: @llvm.floor.v2f64(<2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.floor.v2f64(<2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrdpim

  vf = __builtin_vsx_xvrspi(vf);
  // CHECK-LABEL: try-xvrspi
  // CHECK-UNCONSTRAINED: @llvm.round.v4f32(<4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.round.v4f32(<4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrspi

  vd = __builtin_vsx_xvrdpi(vd);
  // CHECK-LABEL: try-xvrdpi
  // CHECK-UNCONSTRAINED: @llvm.round.v2f64(<2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.round.v2f64(<2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrdpi

  vf = __builtin_vsx_xvrspic(vf);
  // CHECK-LABEL: try-xvrspic
  // CHECK-UNCONSTRAINED: @llvm.rint.v4f32(<4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.rint.v4f32(<4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrspic

  vd = __builtin_vsx_xvrdpic(vd);
  // CHECK-LABEL: try-xvrdpic
  // CHECK-UNCONSTRAINED: @llvm.rint.v2f64(<2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.rint.v2f64(<2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrdpic

  vf = __builtin_vsx_xvrspip(vf);
  // CHECK-LABEL: try-xvrspip
  // CHECK-UNCONSTRAINED: @llvm.ceil.v4f32(<4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.ceil.v4f32(<4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrspip

  vd = __builtin_vsx_xvrdpip(vd);
  // CHECK-LABEL: try-xvrdpip
  // CHECK-UNCONSTRAINED: @llvm.ceil.v2f64(<2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.ceil.v2f64(<2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrdpip

  vf = __builtin_vsx_xvrspiz(vf);
  // CHECK-LABEL: try-xvrspiz
  // CHECK-UNCONSTRAINED: @llvm.trunc.v4f32(<4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.trunc.v4f32(<4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrspiz

  vd = __builtin_vsx_xvrdpiz(vd);
  // CHECK-LABEL: try-xvrdpiz
  // CHECK-UNCONSTRAINED: @llvm.trunc.v2f64(<2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.trunc.v2f64(<2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvrdpiz

  vf = __builtin_vsx_xvmaddasp(vf, vf, vf);
  // CHECK-LABEL: try-xvmaddasp
  // CHECK-UNCONSTRAINED: @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvmaddasp

  vd = __builtin_vsx_xvmaddadp(vd, vd, vd);
  // CHECK-LABEL: try-xvmaddadp
  // CHECK-UNCONSTRAINED: @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-CONSTRAINED: @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvmaddadp

  vf = __builtin_vsx_xvnmaddasp(vf, vf, vf);
  // CHECK-LABEL: try-xvnmaddasp
  // CHECK-UNCONSTRAINED: [[RESULT:%[^ ]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-UNCONSTRAINED: fneg <4 x float> [[RESULT]]
  // CHECK-CONSTRAINED: [[RESULT:%[^ ]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-CONSTRAINED: fneg <4 x float> [[RESULT]]
  // NOT-FIXME-CHECK: xvnmaddasp
  // FIXME-CHECK: xvmaddasp
  // FIXME-CHECK: xvnegsp

  vd = __builtin_vsx_xvnmaddadp(vd, vd, vd);
  // CHECK-LABEL: try-xvnmaddadp
  // CHECK-UNCONSTRAINED: [[RESULT:%[^ ]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-UNCONSTRAINED: fneg <2 x double> [[RESULT]]
  // CHECK-CONSTRAINED: [[RESULT:%[^ ]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-CONSTRAINED: fneg <2 x double> [[RESULT]]
  // CHECK-ASM: xvnmaddadp

  vf = __builtin_vsx_xvmsubasp(vf, vf, vf);
  // CHECK-LABEL: try-xvmsubasp
  // CHECK-UNCONSTRAINED: [[RESULT:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK-UNCONSTRAINED: @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[RESULT]])
  // CHECK-CONSTRAINED: [[RESULT:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK-CONSTRAINED: @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[RESULT]]) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM: xvmsubasp

  vd = __builtin_vsx_xvmsubadp(vd, vd, vd);
  // CHECK-LABEL: try-xvmsubadp
  // CHECK-UNCONSTRAINED: [[RESULT:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK-UNCONSTRAINED: @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[RESULT]])
  // CHECK-CONSTRAINED: [[RESULT:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK-CONSTRAINED: @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[RESULT]]) #{{[0-9]+}} [ "fp.control"(metadata !"rte") ]
  // CHECK-ASM:  xvmsubadp

  vf = __builtin_vsx_xvnmsubasp(vf, vf, vf);
  // CHECK-LABEL: try-xvnmsubasp
  // CHECK-UNCONSTRAINED: call <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-CONSTRAINED: call <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: xvnmsubasp

  vd = __builtin_vsx_xvnmsubadp(vd, vd, vd);
  // CHECK-LABEL: try-xvnmsubadp
  // CHECK-UNCONSTRAINED: call <2 x double> @llvm.ppc.fnmsub.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-CONSTRAINED: call <2 x double> @llvm.ppc.fnmsub.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: xvnmsubadp
}
