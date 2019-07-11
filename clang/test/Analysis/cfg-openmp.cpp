// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG %s 2>&1 -fopenmp | FileCheck %s

// CHECK-LABEL:  void xxx(int argc)
void xxx(int argc) {
// CHECK:        [B1]
// CHECK-NEXT:   1: int x;
  int x;
// CHECK-NEXT:   2: x
// CHECK-NEXT:   3: [B1.2] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   4: argc
// CHECK-NEXT:   5: [B1.4] = [B1.3]
// CHECK-NEXT:   6: #pragma omp atomic read
// CHECK-NEXT:    [B1.5];
#pragma omp atomic read
  argc = x;
// CHECK-NEXT:   7: x
// CHECK-NEXT:   8: [B1.7] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   9: argc
// CHECK-NEXT:  10: [B1.9] = [B1.8]
// CHECK-NEXT:  11: #pragma omp critical
// CHECK-NEXT:    [B1.10];
#pragma omp critical
  argc = x;
// CHECK-NEXT:  12: x
// CHECK-NEXT:  13: [B1.12] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  14: argc
// CHECK-NEXT:  15: [B1.14] = [B1.13]
// CHECK-NEXT:  16: #pragma omp distribute parallel for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.15];
#pragma omp distribute parallel for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  17: x
// CHECK-NEXT:  18: [B1.17] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  19: argc
// CHECK-NEXT:  20: [B1.19] = [B1.18]
// CHECK-NEXT:  21: #pragma omp distribute parallel for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.20];
#pragma omp distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  22: x
// CHECK-NEXT:  23: [B1.22] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  24: argc
// CHECK-NEXT:  25: [B1.24] = [B1.23]
// CHECK-NEXT:  26: #pragma omp distribute simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.25];
#pragma omp distribute simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  27: x
// CHECK-NEXT:  28: [B1.27] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  29: argc
// CHECK-NEXT:  30: [B1.29] = [B1.28]
// CHECK-NEXT:  31: #pragma omp for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.30];
#pragma omp for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  32: x
// CHECK-NEXT:  33: [B1.32] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  34: argc
// CHECK-NEXT:  35: [B1.34] = [B1.33]
// CHECK-NEXT:  36: #pragma omp for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.35];
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  37: x
// CHECK-NEXT:  38: [B1.37] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  39: argc
// CHECK-NEXT:  40: [B1.39] = [B1.38]
// CHECK-NEXT:  41: #pragma omp master
// CHECK-NEXT:    [B1.40];
#pragma omp master
  argc = x;
// CHECK-NEXT:  42: x
// CHECK-NEXT:  43: [B1.42] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  44: argc
// CHECK-NEXT:  45: [B1.44] = [B1.43]
// CHECK-NEXT:  46: #pragma omp ordered
// CHECK-NEXT:    [B1.45];
// CHECK-NEXT:  47: #pragma omp for ordered
// CHECK-NEXT:    for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:[B1.46]    }
#pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered
    argc = x;
  }
// CHECK-NEXT:  48: x
// CHECK-NEXT:  49: [B1.48] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  50: argc
// CHECK-NEXT:  51: [B1.50] = [B1.49]
// CHECK-NEXT:  52: #pragma omp parallel for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.51];
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  53: x
// CHECK-NEXT:  54: [B1.53] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  55: argc
// CHECK-NEXT:  56: [B1.55] = [B1.54]
// CHECK-NEXT:  57: #pragma omp parallel for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.56];
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  58: x
// CHECK-NEXT:  59: [B1.58] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  60: argc
// CHECK-NEXT:  61: [B1.60] = [B1.59]
// CHECK-NEXT:  62: #pragma omp parallel
// CHECK-NEXT:    [B1.61];
#pragma omp parallel
  argc = x;
// CHECK-NEXT:  63: x
// CHECK-NEXT:  64: [B1.63] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  65: argc
// CHECK-NEXT:  66: [B1.65] = [B1.64]
// CHECK-NEXT:  67: #pragma omp parallel sections
// CHECK-NEXT:    {
// CHECK-NEXT:        [B1.66];
// CHECK-NEXT:    }
#pragma omp parallel sections
  {
    argc = x;
  }
// CHECK-NEXT:  68: x
// CHECK-NEXT:  69: [B1.68] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  70: argc
// CHECK-NEXT:  71: [B1.70] = [B1.69]
// CHECK-NEXT:  72: #pragma omp simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.71];
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  73: x
// CHECK-NEXT:  74: [B1.73] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  75: argc
// CHECK-NEXT:  76: [B1.75] = [B1.74]
// CHECK-NEXT:  77: #pragma omp single
// CHECK-NEXT:    [B1.76];
#pragma omp single
  argc = x;
// CHECK-NEXT:  78: x
// CHECK-NEXT:  79: [B1.78] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  80: argc
// CHECK-NEXT:  81: [B1.80] = [B1.79]
// CHECK-NEXT:  82: #pragma omp target depend(in : argc)
// CHECK-NEXT:    [B1.81];
#pragma omp target depend(in \
                          : argc)
  argc = x;
// CHECK-NEXT:  83: x
// CHECK-NEXT:  84: [B1.83] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  85: argc
// CHECK-NEXT:  86: [B1.85] = [B1.84]
// CHECK-NEXT:  87: #pragma omp target parallel for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.86];
#pragma omp target parallel for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  88: x
// CHECK-NEXT:  89: [B1.88] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  90: argc
// CHECK-NEXT:  91: [B1.90] = [B1.89]
// CHECK-NEXT:  92: #pragma omp target parallel for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.91];
#pragma omp target parallel for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  93: x
// CHECK-NEXT:  94: [B1.93] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  95: argc
// CHECK-NEXT:  96: [B1.95] = [B1.94]
// CHECK-NEXT:  97: #pragma omp target parallel
// CHECK-NEXT:    [B1.96];
#pragma omp target parallel
  argc = x;
// CHECK-NEXT:  98: x
// CHECK-NEXT:  99: [B1.98] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 100: argc
// CHECK-NEXT: 101: [B1.100] = [B1.99]
// CHECK-NEXT: 102: #pragma omp target simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.101];
#pragma omp target simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 103: x
// CHECK-NEXT: 104: [B1.103] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 105: argc
// CHECK-NEXT: 106: [B1.105] = [B1.104]
// CHECK-NEXT: 107: #pragma omp target teams distribute
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.106];
#pragma omp target teams distribute
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 108: x
// CHECK-NEXT: 109: [B1.108] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 110: argc
// CHECK-NEXT: 111: [B1.110] = [B1.109]
// CHECK-NEXT: 112: #pragma omp target teams distribute parallel for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.111];
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 113: x
// CHECK-NEXT: 114: [B1.113] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 115: argc
// CHECK-NEXT: 116: [B1.115] = [B1.114]
// CHECK-NEXT: 117: #pragma omp target teams distribute parallel for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.116];
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 118: x
// CHECK-NEXT: 119: [B1.118] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 120: argc
// CHECK-NEXT: 121: [B1.120] = [B1.119]
// CHECK-NEXT: 122: #pragma omp target teams distribute simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.121];
#pragma omp target teams distribute simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 123: x
// CHECK-NEXT: 124: [B1.123] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 125: argc
// CHECK-NEXT: 126: [B1.125] = [B1.124]
// CHECK-NEXT: 127: #pragma omp target teams
// CHECK-NEXT:    [B1.126];
#pragma omp target teams
  argc = x;
// CHECK-NEXT: 128: #pragma omp target update to(x)
#pragma omp target update to(x)
// CHECK-NEXT: 129: x
// CHECK-NEXT: 130: [B1.129] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 131: argc
// CHECK-NEXT: 132: [B1.131] = [B1.130]
  argc = x;
// CHECK-NEXT: 133: x
// CHECK-NEXT: 134: [B1.133] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 135: argc
// CHECK-NEXT: 136: [B1.135] = [B1.134]
// CHECK-NEXT: 137: #pragma omp task
// CHECK-NEXT:    [B1.136];
#pragma omp task
  argc = x;
// CHECK-NEXT: 138: x
// CHECK-NEXT: 139: [B1.138] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 140: argc
// CHECK-NEXT: 141: [B1.140] = [B1.139]
// CHECK-NEXT: 142: #pragma omp taskgroup
// CHECK-NEXT:    [B1.141];
#pragma omp taskgroup
  argc = x;
// CHECK-NEXT: 143: x
// CHECK-NEXT: 144: [B1.143] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 145: argc
// CHECK-NEXT: 146: [B1.145] = [B1.144]
// CHECK-NEXT: 147: #pragma omp taskloop
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.146];
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 148: x
// CHECK-NEXT: 149: [B1.148] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 150: argc
// CHECK-NEXT: 151: [B1.150] = [B1.149]
// CHECK-NEXT: 152: #pragma omp taskloop simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.151];
#pragma omp taskloop simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT: 153: x
// CHECK-NEXT: 154: [B1.153] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 155: argc
// CHECK-NEXT: 156: [B1.155] = [B1.154]
// CHECK-NEXT: 157: #pragma omp teams distribute parallel for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.156];
// CHECK-NEXT: 158: #pragma omp target
#pragma omp target
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:[B1.157] 159: x
// CHECK-NEXT: 160: [B1.159] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 161: argc
// CHECK-NEXT: 162: [B1.161] = [B1.160]
// CHECK-NEXT: 163: #pragma omp teams distribute parallel for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.162];
// CHECK-NEXT: 164: #pragma omp target
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:[B1.163] 165: x
// CHECK-NEXT: 166: [B1.165] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 167: argc
// CHECK-NEXT: 168: [B1.167] = [B1.166]
// CHECK-NEXT: 169: #pragma omp teams distribute simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.168];
// CHECK-NEXT: 170: #pragma omp target
#pragma omp target
#pragma omp teams distribute simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:[B1.169] 171: x
// CHECK-NEXT: 172: [B1.171] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: 173: argc
// CHECK-NEXT: 174: [B1.173] = [B1.172]
// CHECK-NEXT: 175: #pragma omp teams
// CHECK-NEXT:    [B1.174];
// CHECK-NEXT: 176: #pragma omp target
#pragma omp target
#pragma omp teams
  argc = x;
// CHECK-NEXT:[B1.175]   Preds
}

