// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++ [dcl.ref]p5:
//   There shall be no references to references, no arrays of
//   references, and no pointers to references.

// The crazy formatting in here is to enforce the exact report locations.

typedef int &intref;
typedef intref &intrefref;

template <class T> class RefMem { // expected-warning{{class 'RefMem<int &>' does not declare any constructor to initialize its non-modifiable members}}
  T
    &
      member; // expected-note{{reference member 'member' will never be initialized}}
};

struct RefRef {
  int
      &
        &             // expected-error {{declared as a reference to a reference}}
          refref0;

  intref
         &
           refref1; // collapses

  intrefref
            &
              refref2; // collapses

  RefMem
        <
         int
            &
             >
               refref3; // collapses expected-note{{in instantiation of template class 'RefMem<int &>' requested here}}
};


template <class T> class PtrMem {
  T
    *                   // expected-error {{declared as a pointer to a reference}}
      member;
};

struct RefPtr {
  typedef
          int
              &
                *       // expected-error {{declared as a pointer to a reference}}
                  intrefptr;

  typedef
          intref
                 *      // expected-error {{declared as a pointer to a reference}}
                   intrefptr2;

  int
      &
        *               // expected-error {{declared as a pointer to a reference}}
          refptr0;

  intref
         *              // expected-error {{declared as a pointer to a reference}}
           refptr1;

  PtrMem
        <
         int
            &
             >
               refptr2; // expected-note {{in instantiation}}
};

template <class T> class ArrMem {
  T
    member
           [ // expected-error {{declared as array of references}}
            10
              ];
};
template <class T, unsigned N> class DepArrMem {
  T
    member
           [ // expected-error {{declared as array of references}}
            N
             ];
};

struct RefArr {
  typedef 
          int
              &
                intrefarr
                         [ // expected-error {{declared as array of references}}
                          2
                           ];

  typedef
          intref
                 intrefarr
                          [ // expected-error {{declared as array of references}}
                           2
                            ];

  int
      &
        refarr0
               [ // expected-error {{declared as array of references}}
                2
                 ];
  intref
         refarr1
                [ // expected-error {{declared as array of references}}
                 2
                  ];
  ArrMem
        <
         int
            &
             >
               refarr2; // expected-note {{in instantiation}}
  DepArrMem
           <
            int
               &,
                  10
                    >
                      refarr3; // expected-note {{in instantiation}}
};


//   The declaration of a reference shall contain an initializer
//   (8.5.3) except when the declaration contains an explicit extern
//   specifier (7.1.1), is a class member (9.2) declaration within a
//   class definition, or is the declaration of a parameter or a
//   return type (8.3.5); see 3.1. A reference shall be initialized to
//   refer to a valid object or function. [ Note: in particular, a
//   null reference cannot exist in a well-defined program, because
//   the only way to create such a reference would be to bind it to
//   the "object" obtained by dereferencing a null pointer, which
//   causes undefined behavior. As described in 9.6, a reference
//   cannot be bound directly to a bit-field.

