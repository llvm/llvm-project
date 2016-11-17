// Merge success
namespace N1 {
  extern int x0;
}

// Merge multiple namespaces
namespace N2 {
  extern int x;
}
namespace N2 {
  extern float y;
}

// Merge namespace with conflict
namespace N3 {
  extern double z;
}
