/// NB: The exact contents of this file are compared against the corresponding
///     "_expected" file in this directory; any changes to this file, including
///     comments, will require updating the expected file.

int main() {
  int Go = 0;
  int Times = 0;
start:
  Times += 1;
  for (int I = 0; I < 3; ++I) {
    if (I == 1)
      Go += 1;
    if (I > 1 && Go % 2)
      goto start; // !dex_label first_goto
  }
  if (Times < 4)
    goto start; // !dex_label second_goto
  return 0;
}

/*
---
? !where {function: main}
: !step 'exactly':
  - 6
  - 7
  - 9
  - 10
  - 11
  - 13
  - 15
  - 10
  - 11
  - 12
  - 13
  - 15
  - 10
  - 11
  - 13
  - 14
  - 9
  - 10
  - 11
  - 13
  - 15
  - 10
  - 11
  - 12
  - 13
  - 15
  - 10
  - 11
  - 13
  - 15
  - 10
  - 16
  - 17
  - 9
  - 10
  - 11
  - 13
  - 15
  - 10
  - 11
  - 12
  - 13
  - 15
  - 10
  - 11
  - 13
  - 14
  - 9
  - 10
  - 11
  - 13
  - 15
  - 10
  - 11
  - 12
  - 13
  - 15
  - 10
  - 11
  - 13
  - 15
  - 10
  - 16
  - 18
  ? !and {lines: !label 'first_goto'}
  : !step 'exactly':
    - !label 'first_goto'
    - !label 'first_goto'
  ? !and {lines: !label 'second_goto'}
  : !step 'exactly':
    - !label 'second_goto'
...
*/
