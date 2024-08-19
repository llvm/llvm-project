// RUN: %clang_cc1 %s -verify -fopenacc -fcxx-exceptions


void ReturnTest() {
#pragma acc parallel
  {
    (void)[]() { return; };
  }

#pragma acc parallel
  {
    try {}
    catch(...){
      return; // expected-error{{invalid return out of OpenACC Compute Construct}}
    }
  }
}

template<typename T>
void BreakContinue() {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    switch(i) {
      case 0:
      break; // leaves switch, not 'for'.
      default:
      i +=2;
      break;
    }
    if (i == 2)
      continue;

    break;  // expected-error{{invalid branch out of OpenACC Compute Construct}}
  }

  int j;
  switch(j) {
    case 0:
#pragma acc parallel
    {
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
    }
    case 1:
#pragma acc parallel
    {
    }
    break;
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    if (i > 1)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  switch(j) {
    case 1:
      break;
  }

#pragma acc parallel
  {
    for(int i = 1; i < 100; i++) {
      if (i > 4)
        break;
    }
  }

  for (int i =0; i < 5; ++i) {
#pragma acc parallel
    {
      continue; // expected-error{{invalid branch out of OpenACC Compute Construct}}
    }
  }

#pragma acc parallel
  for (int i =0; i < 5; ++i) {
    continue;
  }

#pragma acc parallel
  for (int i =0; i < 5; ++i) {
    {
      continue;
    }
  }

  for (int i =0; i < 5; ++i) {
#pragma acc parallel
    {
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
    }
  }

#pragma acc parallel
  while (j) {
    --j;
    if (j > 4)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  do {
    --j;
    if (j > 4)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
  } while (j );
}

template<typename T>
void DuffsDevice() {
  int j;
  switch (j) {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    case 0: // expected-error{{invalid branch into OpenACC Compute Construct}}
      {}
  }
  }

  switch (j) {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    default: // expected-error{{invalid branch into OpenACC Compute Construct}}
      {}
  }
  }

  switch (j) {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    case 'a' ... 'z': // expected-error{{invalid branch into OpenACC Compute Construct}}
      {}
  }
  }
}

void Exceptions() {
#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    throw 5; // expected-error{{invalid throw out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    throw; // expected-error{{invalid throw out of OpenACC Compute Construct}}
  }

#pragma acc serial
  for(int i = 0; i < 5; ++i) {
    throw; // expected-error{{invalid throw out of OpenACC Compute Construct}}
  }

#pragma acc kernels
  for(int i = 0; i < 5; ++i) {
    throw; // expected-error{{invalid throw out of OpenACC Compute Construct}}
  }


#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    try {
    throw 5;
    } catch(float f) {
    }
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    try {
    throw 5;
    } catch(int f) {
    }
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    try {
    throw 5;
    } catch(...) {
    }
  }
#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    try {
    throw;
    } catch(...) {
    }
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    try {
    throw;
    } catch(...) {
      throw; // expected-error{{invalid throw out of OpenACC Compute Construct}}
    }
  }
#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    try {
    throw;
    } catch(int f) {
      throw; // expected-error{{invalid throw out of OpenACC Compute Construct}}
    }
  }
}

void Instantiate() {
  BreakContinue<int>();
  DuffsDevice<int>();
}
