// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -o - -fsyntax-only %s -verify

// This file is based on tools/clang/test/SemaHLSL/hlsl/202x-cbuffer-contexts.hlsl
// from the microsoft/DirectXShaderCompiler repo.

cbuffer A {
  int Y;
  // expected-error@+1 {{invalid declaration inside cbuffer}}
  namespace N {
  }
  float4 F5;
}

cbuffer A {
  // expected-error@+1 {{invalid declaration inside cbuffer}}
  cbuffer Nested {
  }
}

tbuffer TB {
  // expected-error@+1 {{invalid declaration inside tbuffer}}
  namespace NS {}
}

tbuffer TB2 {
  // expected-error@+1{{invalid declaration inside tbuffer}}
  cbuffer CB2 {
    int X;
  }
}

namespace Valid {
  cbuffer CBValid {
    int CompletelyFine;
  }

  tbuffer TBValid {
    float StillFine;
  }

  cbuffer GoingOffTheRails {
    // expected-error@+1{{invalid declaration inside cbuffer}}
    tbuffer NotCool {
      ; // even if it is empty...
    }
  }

  cbuffer GoingOffTheRailsAgain {
    // expected-error@+1{{invalid declaration inside cbuffer}}
    tbuffer StillNotCool {
      // expected-error@+1{{invalid declaration inside tbuffer}}
      cbuffer Turtles {
        // expected-error@+1{{invalid declaration inside cbuffer}}
        tbuffer AllTheWayDown {
          ;
        }
      }
    }
  }
}
