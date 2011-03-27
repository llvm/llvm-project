// RUN: %clang_cc1 -fsyntax-only -verify %s

//PR9463
int subfun(const char *text) {
  const char *tmp = text;
  return 0;
}

void fun(const char* text) {
  int count = 0;
  bool check = true;

  if (check)
    {
      const char *end = text;

      if (check)
        {
          do
            {
              if (check)
                {
                  count = subfun(end);
                  goto end;
                }

              check = !check;
            }
          while (check);
        }
      // also works, after commenting following line of source code
      int e = subfun(end);
    }
 end:
  if (check)
    ++count;
}

const char *text = "some text";

int main() {
	const char *ptr = text;

	fun(ptr);

	return 0;
}
