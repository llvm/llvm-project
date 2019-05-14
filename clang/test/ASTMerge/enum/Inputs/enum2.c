// Matching
enum E1 {
  E1Enumerator1,
  E1Enumerator2 = 3,
  E1Enumerator3
} x1;

// Value mismatch
enum E2 {
  E2Enumerator1,
  E2Enumerator2 = 4,
  E2Enumerator3
} x2;

// Name mismatch
enum E3 {
  E3Enumerator1,
  E3Enumerator = 3,
  E3Enumerator3
} x3;

// Missing enumerator
enum E4 {
  E4Enumerator1,
  E4Enumerator2
} x4;

// Extra enumerator
enum E5 {
  E5Enumerator1,
  E5Enumerator2,
  E5Enumerator3,
  E5Enumerator4
} x5;

// Matching, with typedef
typedef enum {
  E6Enumerator1,
  E6Enumerator2
} E6;

E6 x6;
