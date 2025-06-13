// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

struct Element {};
struct Container {};

Element *begin(Container &);
Element *end(Container &);

void for_range() {
  Container c;
  for (Element &e : c)
    ;
}

// CIR: cir.func @_Z5beginR9Container(!cir.ptr<!rec_Container>) -> !cir.ptr<!rec_Element>
// CIR: cir.func @_Z3endR9Container(!cir.ptr<!rec_Container>) -> !cir.ptr<!rec_Element

// CIR: cir.func @_Z9for_rangev()
// CIR:    %[[C_ADDR:.*]] = cir.alloca !rec_Container{{.*}} ["c"]
// CIR:    cir.scope {
// CIR:      %[[RANGE_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Container>{{.*}} ["__range1", init, const]
// CIR:      %[[BEGIN_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["__begin1", init]
// CIR:      %[[END_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["__end1", init]
// CIR:      %[[E_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["e", init, const]
// CIR:      cir.store{{.*}} %[[C_ADDR]], %[[RANGE_ADDR]]
// CIR:      %[[C_REF:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:      %[[BEGIN:.*]] = cir.call @_Z5beginR9Container(%[[C_REF]])
// CIR:      cir.store{{.*}} %[[BEGIN]], %[[BEGIN_ADDR]]
// CIR:      %[[C_REF2:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:      %[[END:.*]] = cir.call @_Z3endR9Container(%[[C_REF2]])
// CIR:      cir.store{{.*}} %[[END]], %[[END_ADDR]]
// CIR:      cir.for : cond {
// CIR:        %[[BEGIN:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:        %[[END:.*]] = cir.load{{.*}} %[[END_ADDR]]
// CIR:        %[[CMP:.*]] = cir.cmp(ne, %[[BEGIN]], %[[END]])
// CIR:        cir.condition(%[[CMP]])
// CIR:      } body {
// CIR:        %[[E:.*]] = cir.load deref{{.*}} %[[BEGIN_ADDR]]
// CIR:        cir.store{{.*}} %[[E]], %[[E_ADDR]]
// CIR:        cir.yield
// CIR:      } step {
// CIR:        %[[BEGIN:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:        %[[STEP:.*]] = cir.const #cir.int<1>
// CIR:        %[[NEXT:.*]] = cir.ptr_stride(%[[BEGIN]] {{.*}}, %[[STEP]] {{.*}})
// CIR:        cir.store{{.*}} %[[NEXT]], %[[BEGIN_ADDR]]
// CIR:        cir.yield
// CIR:      }
// CIR:    }

struct C2 {
  Element *begin();
  Element *end();
};

void for_range2() {
  C2 c;
  for (Element &e : c)
    ;
}

// CIR: cir.func @_Z10for_range2v()
// CIR:    %[[C_ADDR:.*]] = cir.alloca !rec_C2{{.*}} ["c"]
// CIR:    cir.scope {
// CIR:      %[[RANGE_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C2>{{.*}} ["__range1", init, const]
// CIR:      %[[BEGIN_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["__begin1", init]
// CIR:      %[[END_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["__end1", init]
// CIR:      %[[E_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["e", init, const]
// CIR:      cir.store{{.*}} %[[C_ADDR]], %[[RANGE_ADDR]]
// CIR:      %[[C_REF:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:      %[[BEGIN:.*]] = cir.call @_ZN2C25beginEv(%[[C_REF]])
// CIR:      cir.store{{.*}} %[[BEGIN]], %[[BEGIN_ADDR]]
// CIR:      %[[C_REF2:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:      %[[END:.*]] = cir.call @_ZN2C23endEv(%[[C_REF2]])
// CIR:      cir.store{{.*}} %[[END]], %[[END_ADDR]]
// CIR:      cir.for : cond {
// CIR:        %[[BEGIN:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:        %[[END:.*]] = cir.load{{.*}} %[[END_ADDR]]
// CIR:        %[[CMP:.*]] = cir.cmp(ne, %[[BEGIN]], %[[END]])
// CIR:        cir.condition(%[[CMP]])
// CIR:      } body {
// CIR:        %[[E:.*]] = cir.load deref{{.*}} %[[BEGIN_ADDR]]
// CIR:        cir.store{{.*}} %[[E]], %[[E_ADDR]]
// CIR:        cir.yield
// CIR:      } step {
// CIR:        %[[BEGIN:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:        %[[STEP:.*]] = cir.const #cir.int<1>
// CIR:        %[[NEXT:.*]] = cir.ptr_stride(%[[BEGIN]] {{.*}}, %[[STEP]] {{.*}})
// CIR:        cir.store{{.*}} %[[NEXT]], %[[BEGIN_ADDR]]
// CIR:        cir.yield
// CIR:      }
// CIR:    }

// Iterator class definition
class Iterator {
public:
  Element& operator*();
  Iterator& operator++();
  bool operator!=(const Iterator& other) const;
};

class C3 {
public:
  Iterator begin();
  Iterator end();
};

void for_range3() {
  C3 c;
  for (Element& e : c)
    ;
}

// CIR: cir.func @_Z10for_range3v()
// CIR:    %[[C_ADDR:.*]] = cir.alloca !rec_C3{{.*}} ["c"]
// CIR:    cir.scope {
// CIR:      %[[RANGE_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C3>{{.*}} ["__range1", init, const]
// CIR:      %[[BEGIN_ADDR:.*]] = cir.alloca !rec_Iterator, !cir.ptr<!rec_Iterator>{{.*}} ["__begin1"]
// CIR:      %[[END_ADDR:.*]] = cir.alloca !rec_Iterator, !cir.ptr<!rec_Iterator>{{.*}} ["__end1"]
// CIR:      %[[E_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Element>{{.*}} ["e", init, const]
// CIR:      cir.store{{.*}} %[[C_ADDR]], %[[RANGE_ADDR]]
// CIR:      cir.for : cond {
// CIR:        %[[ITER_NE:.*]] = cir.call @_ZNK8IteratorneERKS_(%[[BEGIN_ADDR]], %[[END_ADDR]])
// CIR:        cir.condition(%[[ITER_NE]])
// CIR:      } body {
// CIR:        %[[E:.*]] = cir.call @_ZN8IteratordeEv(%[[BEGIN_ADDR]])
// CIR:        cir.store{{.*}} %[[E]], %[[E_ADDR]]
// CIR:        cir.yield
// CIR:      } step {
// CIR:        %[[ITER_NEXT:.*]] = cir.call @_ZN8IteratorppEv(%[[BEGIN_ADDR]])
// CIR:        cir.yield
// CIR:      }
// CIR:    }
