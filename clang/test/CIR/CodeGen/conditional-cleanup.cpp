// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.eh.cir
// RUN: FileCheck --check-prefix=CIR_EH --input-file=%t.eh.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat %s -o %t.eh.flat.cir
// RUN: FileCheck --check-prefix=CIR_FLAT_EH --input-file=%t.eh.flat.cir %s

typedef __typeof(sizeof(0)) size_t;

// Declare the reserved global placement new.
void *operator new(size_t, void*);

namespace test7 {
  struct A { A(); ~A(); };
  struct B {
    static void *operator new(size_t size) throw();
    B(const A&, B*);
    ~B();
  };

  B *test() {
    return new B(A(), new B(A(), 0));
  }
}

// CIR-DAG: ![[A:.*]] = !cir.struct<struct "test7::A" {!u8i}
// CIR-DAG: ![[B:.*]] = !cir.struct<struct "test7::B" {!u8i}

// CIR_EH-DAG: ![[A:.*]] = !cir.struct<struct "test7::A" {!u8i}
// CIR_EH-DAG: ![[B:.*]] = !cir.struct<struct "test7::B" {!u8i}

// CIR-LABEL: _ZN5test74testEv
// CIR:   %[[RET_VAL:.*]] = cir.alloca !cir.ptr<![[B]]>, !cir.ptr<!cir.ptr<![[B]]>>, ["__retval"] {alignment = 8 : i64}
// CIR:   cir.scope {
// CIR:     %[[TMP_A0:.*]] = cir.alloca ![[A]], !cir.ptr<![[A]]>, ["ref.tmp0"] {alignment = 1 : i64}
// CIR:     %[[CLEANUP_COND_OUTER:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR:     %[[TMP_A1:.*]] = cir.alloca ![[A]], !cir.ptr<![[A]]>, ["ref.tmp1"] {alignment = 1 : i64}
// CIR:     %[[CLEANUP_COND_INNER:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR:     %[[FALSE0:.*]] = cir.const #false
// CIR:     %[[TRUE0:.*]] = cir.const #true
// CIR:     %[[FALSE1:.*]] = cir.const #false
// CIR:     %[[TRUE1:.*]] = cir.const #true

// CIR:     %[[NULL_CHECK0:.*]] = cir.cmp(ne
// CIR:     %[[PTR_B0:.*]] = cir.cast(bitcast
// CIR:     cir.store align(1) %[[FALSE1]], %[[CLEANUP_COND_OUTER]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.store align(1) %[[FALSE0]], %[[CLEANUP_COND_INNER]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.if %[[NULL_CHECK0]] {

// Ctor call: @test7::A::A()
// CIR:       cir.call @_ZN5test71AC1Ev(%[[TMP_A0]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:       cir.store %[[TRUE1]], %[[CLEANUP_COND_OUTER]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR:       %[[NULL_CHECK1:.*]] = cir.cmp(ne
// CIR:       %[[PTR_B1:.*]] = cir.cast(bitcast
// CIR:       cir.if %[[NULL_CHECK1]] {

// Ctor call: @test7::A::A()
// CIR:         cir.call @_ZN5test71AC1Ev(%[[TMP_A1]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:         cir.store %[[TRUE0]], %[[CLEANUP_COND_INNER]] : !cir.bool, !cir.ptr<!cir.bool>
// Ctor call: @test7::B::B()
// CIR:         cir.call @_ZN5test71BC1ERKNS_1AEPS0_(%[[PTR_B1]], %[[TMP_A1]], {{.*}}) : (!cir.ptr<![[B]]>, !cir.ptr<![[A]]>, !cir.ptr<![[B]]>) -> ()
// CIR:       }

// Ctor call: @test7::B::B()
// CIR:       cir.call @_ZN5test71BC1ERKNS_1AEPS0_(%[[PTR_B0]], %[[TMP_A0]], %[[PTR_B1]]) : (!cir.ptr<![[B]]>, !cir.ptr<![[A]]>, !cir.ptr<![[B]]>) -> ()
// CIR:     }
// CIR:     cir.store %[[PTR_B0]], %[[RET_VAL]] : !cir.ptr<![[B]]>, !cir.ptr<!cir.ptr<![[B]]>>
// CIR:     %[[DO_CLEANUP_INNER:.*]] = cir.load %[[CLEANUP_COND_INNER]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[DO_CLEANUP_INNER]] {
// Dtor call: @test7::A::~A()
// CIR:       cir.call @_ZN5test71AD1Ev(%[[TMP_A1]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:     }
// CIR:     %[[DO_CLEANUP_OUTER:.*]] = cir.load %[[CLEANUP_COND_OUTER]] : !cir.ptr<!cir.bool>, !cir.bool
// Dtor call: @test7::A::~A()
// CIR:     cir.if %[[DO_CLEANUP_OUTER]] {
// CIR:       cir.call @_ZN5test71AD1Ev(%[[TMP_A0]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// CIR_EH-DAG: #[[$ATTR_0:.+]] = #cir.bool<false> : !cir.bool
// CIR_EH-DAG: #[[$ATTR_1:.+]] = #cir<extra({nothrow = #cir.nothrow})>
// CIR_EH-DAG: #[[$ATTR_2:.+]] = #cir<extra({inline = #cir.inline<no>, optnone = #cir.optnone})>
// CIR_EH-DAG: #[[$ATTR_3:.+]] = #cir.bool<true> : !cir.bool

// CIR_EH-LABEL: @_ZN5test74testEv
// CIR_EH:           %[[VAL_0:.*]] = cir.alloca !cir.ptr<!ty_test73A3AB>, !cir.ptr<!cir.ptr<!ty_test73A3AB>>, ["__retval"] {alignment = 8 : i64}
// CIR_EH:           cir.scope {
// CIR_EH:             %[[VAL_1:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR_EH:             %[[VAL_2:.*]] = cir.alloca !ty_test73A3AA, !cir.ptr<!ty_test73A3AA>, ["ref.tmp0"] {alignment = 1 : i64}
// CIR_EH:             %[[VAL_3:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR_EH:             %[[VAL_4:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR_EH:             %[[VAL_5:.*]] = cir.alloca !ty_test73A3AA, !cir.ptr<!ty_test73A3AA>, ["ref.tmp1"] {alignment = 1 : i64}
// CIR_EH:             %[[VAL_6:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR_EH:             %[[VAL_7:.*]] = cir.const #[[$ATTR_0]]
// CIR_EH:             %[[VAL_8:.*]] = cir.const #[[$ATTR_3]]
// CIR_EH:             %[[VAL_9:.*]] = cir.const #[[$ATTR_0]]
// CIR_EH:             %[[VAL_10:.*]] = cir.const #[[$ATTR_3]]
// CIR_EH:             %[[VAL_11:.*]] = cir.const #[[$ATTR_0]]
// CIR_EH:             %[[VAL_12:.*]] = cir.const #[[$ATTR_3]]
// CIR_EH:             %[[VAL_13:.*]] = cir.const #[[$ATTR_0]]
// CIR_EH:             %[[VAL_14:.*]] = cir.const #[[$ATTR_3]]
// CIR_EH:             %[[VAL_15:.*]] = cir.const #{{.*}}<1> : !u64i
// CIR_EH:             %[[VAL_16:.*]] = cir.call @_ZN5test71BnwEm(%[[VAL_15]]) : (!u64i) -> !cir.ptr<!void>
// CIR_EH:             %[[VAL_17:.*]] = cir.const #{{.*}}<null> : !cir.ptr<!void>
// CIR_EH:             %[[VAL_18:.*]] = cir.cmp(ne, %[[VAL_16]], %[[VAL_17]]) : !cir.ptr<!void>, !cir.bool
// CIR_EH:             %[[VAL_19:.*]] = cir.cast(bitcast, %[[VAL_16]] : !cir.ptr<!void>), !cir.ptr<!ty_test73A3AB>
// CIR_EH:             cir.store align(1) %[[VAL_13]], %[[VAL_1]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:             cir.store align(1) %[[VAL_11]], %[[VAL_3]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:             cir.store align(1) %[[VAL_9]], %[[VAL_4]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:             cir.store align(1) %[[VAL_7]], %[[VAL_6]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:             cir.if %[[VAL_18]] {
// CIR_EH:               cir.store %[[VAL_14]], %[[VAL_1]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:               cir.try synthetic cleanup {
// CIR_EH:                 cir.call exception @_ZN5test71AC1Ev(%[[VAL_2]]) : (!cir.ptr<!ty_test73A3AA>) -> () cleanup {
// CIR_EH:                   %[[VAL_20:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                   cir.if %[[VAL_20]] {
// CIR_EH:                     cir.call @_ZdlPvm(%[[VAL_16]], %[[VAL_15]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                   }
// CIR_EH:                   cir.yield
// CIR_EH:                 }
// CIR_EH:                 cir.yield
// CIR_EH:               } catch [#{{.*}} {
// CIR_EH:                 cir.resume
// CIR_EH:               }]
// CIR_EH:               cir.store %[[VAL_12]], %[[VAL_3]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:               %[[VAL_21:.*]] = cir.const #{{.*}}<1> : !u64i
// CIR_EH:               %[[VAL_22:.*]] = cir.call @_ZN5test71BnwEm(%[[VAL_21]]) : (!u64i) -> !cir.ptr<!void>
// CIR_EH:               %[[VAL_23:.*]] = cir.const #{{.*}}<null> : !cir.ptr<!void>
// CIR_EH:               %[[VAL_24:.*]] = cir.cmp(ne, %[[VAL_22]], %[[VAL_23]]) : !cir.ptr<!void>, !cir.bool
// CIR_EH:               %[[VAL_25:.*]] = cir.cast(bitcast, %[[VAL_22]] : !cir.ptr<!void>), !cir.ptr<!ty_test73A3AB>
// CIR_EH:               cir.if %[[VAL_24]] {
// CIR_EH:                 cir.store %[[VAL_10]], %[[VAL_4]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:                 cir.try synthetic cleanup {
// CIR_EH:                   cir.call exception @_ZN5test71AC1Ev(%[[VAL_5]]) : (!cir.ptr<!ty_test73A3AA>) -> () cleanup {
// CIR_EH:                     %[[VAL_26:.*]] = cir.load %[[VAL_4]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_26]] {
// CIR_EH:                       cir.call @_ZdlPvm(%[[VAL_22]], %[[VAL_21]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                     }
// CIR_EH:                     %[[VAL_27:.*]] = cir.load %[[VAL_3]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_27]] {
// CIR_EH:                       cir.call @_ZN5test71AD1Ev(%[[VAL_2]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:                     }
// CIR_EH:                     %[[VAL_28:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_28]] {
// CIR_EH:                       cir.call @_ZdlPvm(%[[VAL_16]], %[[VAL_15]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                     }
// CIR_EH:                     cir.yield
// CIR_EH:                   }
// CIR_EH:                   cir.yield
// CIR_EH:                 } catch [#{{.*}} {
// CIR_EH:                   cir.resume
// CIR_EH:                 }]
// CIR_EH:                 cir.store %[[VAL_8]], %[[VAL_6]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:                 %[[VAL_29:.*]] = cir.const #{{.*}}<null> : !cir.ptr<!ty_test73A3AB>
// CIR_EH:                 cir.try synthetic cleanup {
// CIR_EH:                   cir.call exception @_ZN5test71BC1ERKNS_1AEPS0_(%[[VAL_25]], %[[VAL_5]], %[[VAL_29]]) : (!cir.ptr<!ty_test73A3AB>, !cir.ptr<!ty_test73A3AA>, !cir.ptr<!ty_test73A3AB>) -> () cleanup {
// CIR_EH:                     %[[VAL_30:.*]] = cir.load %[[VAL_6]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_30]] {
// CIR_EH:                       cir.call @_ZN5test71AD1Ev(%[[VAL_5]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:                     }
// CIR_EH:                     %[[VAL_31:.*]] = cir.load %[[VAL_4]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_31]] {
// CIR_EH:                       cir.call @_ZdlPvm(%[[VAL_22]], %[[VAL_21]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                     }
// CIR_EH:                     %[[VAL_32:.*]] = cir.load %[[VAL_3]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_32]] {
// CIR_EH:                       cir.call @_ZN5test71AD1Ev(%[[VAL_2]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:                     }
// CIR_EH:                     %[[VAL_33:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                     cir.if %[[VAL_33]] {
// CIR_EH:                       cir.call @_ZdlPvm(%[[VAL_16]], %[[VAL_15]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                     }
// CIR_EH:                     cir.yield
// CIR_EH:                   }
// CIR_EH:                   cir.yield
// CIR_EH:                 } catch [#{{.*}} {
// CIR_EH:                   cir.resume
// CIR_EH:                 }]
// CIR_EH:                 %[[VAL_34:.*]] = cir.const #[[$ATTR_0]]
// CIR_EH:                 cir.store %[[VAL_34]], %[[VAL_4]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:               }
// CIR_EH:               cir.try synthetic cleanup {
// CIR_EH:                 cir.call exception @_ZN5test71BC1ERKNS_1AEPS0_(%[[VAL_19]], %[[VAL_2]], %[[VAL_25]]) : (!cir.ptr<!ty_test73A3AB>, !cir.ptr<!ty_test73A3AA>, !cir.ptr<!ty_test73A3AB>) -> () cleanup {
// CIR_EH:                   %[[VAL_35:.*]] = cir.load %[[VAL_6]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                   cir.if %[[VAL_35]] {
// CIR_EH:                     cir.call @_ZN5test71AD1Ev(%[[VAL_5]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:                   }
// CIR_EH:                   %[[VAL_36:.*]] = cir.load %[[VAL_4]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                   cir.if %[[VAL_36]] {
// CIR_EH:                     cir.call @_ZdlPvm(%[[VAL_22]], %[[VAL_21]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                   }
// CIR_EH:                   %[[VAL_37:.*]] = cir.load %[[VAL_3]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                   cir.if %[[VAL_37]] {
// CIR_EH:                     cir.call @_ZN5test71AD1Ev(%[[VAL_2]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:                   }
// CIR_EH:                   %[[VAL_38:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:                   cir.if %[[VAL_38]] {
// CIR_EH:                     cir.call @_ZdlPvm(%[[VAL_16]], %[[VAL_15]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:                   }
// CIR_EH:                   cir.yield
// CIR_EH:                 }
// CIR_EH:                 cir.yield
// CIR_EH:               } catch [#{{.*}} {
// CIR_EH:                 cir.resume
// CIR_EH:               }]
// CIR_EH:               %[[VAL_39:.*]] = cir.const #[[$ATTR_0]]
// CIR_EH:               cir.store %[[VAL_39]], %[[VAL_1]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR_EH:             }
// CIR_EH:             cir.store %[[VAL_19]], %[[VAL_0]] : !cir.ptr<!ty_test73A3AB>, !cir.ptr<!cir.ptr<!ty_test73A3AB>>
// CIR_EH:             %[[VAL_40:.*]] = cir.load %[[VAL_6]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:             cir.if %[[VAL_40]] {
// CIR_EH:               cir.call @_ZN5test71AD1Ev(%[[VAL_5]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:             }
// CIR_EH:             %[[VAL_41:.*]] = cir.load %[[VAL_3]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:             cir.if %[[VAL_41]] {
// CIR_EH:               cir.call @_ZN5test71AD1Ev(%[[VAL_2]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:             }
// CIR_EH:             %[[VAL_42:.*]] = cir.load %[[VAL_3]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:             cir.if %[[VAL_42]] {
// CIR_EH:               cir.call @_ZN5test71AD1Ev(%[[VAL_2]]) : (!cir.ptr<!ty_test73A3AA>) -> ()
// CIR_EH:             }
// CIR_EH:             %[[VAL_43:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR_EH:             cir.if %[[VAL_43]] {
// CIR_EH:               cir.call @_ZdlPvm(%[[VAL_16]], %[[VAL_15]]) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_EH:             }
// CIR_EH:           }
// CIR_EH:           %[[VAL_44:.*]] = cir.load %[[VAL_0]] : !cir.ptr<!cir.ptr<!ty_test73A3AB>>, !cir.ptr<!ty_test73A3AB>
// CIR_EH:           cir.return %[[VAL_44]] : !cir.ptr<!ty_test73A3AB>
// CIR_EH:         }

// Nothing special, just test it passes!
// CIR_FLAT_EH-LABEL: @_ZN5test74testEv