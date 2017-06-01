// CUDA struct types with interesting initialization properties.
// Keep in sync with ../CodeGenCUDA/Inputs/cuda-initializers.h.

// Base classes with different initializer variants.

// trivial constructor -- allowed
struct T {
  int t;
};

// empty constructor
struct EC {
  int ec;
  __device__ EC() {}     // -- allowed
  __device__ EC(int) {}  // -- not allowed
};

// empty destructor
struct ED {
  __device__ ~ED() {}     // -- allowed
};

struct ECD {
  __device__ ECD() {}     // -- allowed
  __device__ ~ECD() {}    // -- allowed
};

// empty templated constructor -- allowed with no arguments
struct ETC {
  template <typename... T> __device__ ETC(T...) {}
};

// undefined constructor -- not allowed
struct UC {
  int uc;
  __device__ UC();
};

// undefined destructor -- not allowed
struct UD {
  int ud;
  __device__ ~UD();
};

// empty constructor w/ initializer list -- not allowed
struct ECI {
  int eci;
  __device__ ECI() : eci(1) {}
};

// non-empty constructor -- not allowed
struct NEC {
  int nec;
  __device__ NEC() { nec = 1; }
};

// non-empty destructor -- not allowed
struct NED {
  int ned;
  __device__ ~NED() { ned = 1; }
};

// no-constructor,  virtual method -- not allowed
struct NCV {
  int ncv;
  __device__ virtual void vm() {}
};

// virtual destructor -- not allowed.
struct VD {
  __device__ virtual ~VD() {}
};

// dynamic in-class field initializer -- not allowed
__device__ int f();
struct NCF {
  int ncf = f();
};

// static in-class field initializer.  NVCC does not allow it, but
// clang generates static initializer for this, so we'll accept it.
// We still can't use it on __shared__ vars as they don't allow *any*
// initializers.
struct NCFS {
  int ncfs = 3;
};

// undefined templated constructor -- not allowed
struct UTC {
  template <typename... T> __device__ UTC(T...);
};

// non-empty templated constructor -- not allowed
struct NETC {
  int netc;
  template <typename... T> __device__ NETC(T...) { netc = 1; }
};

// Regular base class -- allowed
struct T_B_T : T {};

// Incapsulated object of allowed class -- allowed
struct T_F_T {
  T t;
};

// array of allowed objects -- allowed
struct T_FA_T {
  T t[2];
};


// Calling empty base class initializer is OK
struct EC_I_EC : EC {
  __device__ EC_I_EC() : EC() {}
};

// .. though passing arguments is not allowed.
struct EC_I_EC1 : EC {
  __device__ EC_I_EC1() : EC(1) {}
};

// Virtual base class -- not allowed
struct T_V_T : virtual T {};

// Inherited from or incapsulated class with non-empty constructor --
// not allowed
struct T_B_NEC : NEC {};
struct T_F_NEC {
  NEC nec;
};
struct T_FA_NEC {
  NEC nec[2];
};


// Inherited from or incapsulated class with non-empty desstructor --
// not allowed
struct T_B_NED : NED {};
struct T_F_NED {
  NED ned;
};
struct T_FA_NED {
  NED ned[2];
};
