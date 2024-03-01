// RUN: %clang_cc1 %s -verify -fopenacc

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

void Return() {
#pragma acc parallel
  {
    return;// expected-error{{invalid return out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  {
    {
      return;// expected-error{{invalid return out of OpenACC Compute Construct}}
    }
  }

#pragma acc parallel
  {
    for (int i = 0; i < 5; ++i) {
      return;// expected-error{{invalid return out of OpenACC Compute Construct}}
    }
  }
}
