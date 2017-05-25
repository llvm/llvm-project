static int static_value = 0;

int
a_function_to_call()
{
    static_value++; // Stop inside the function here.
    return static_value;
}

int second_function(int x){
  for(int i=0; i<10; ++i) {
    a_function_to_call();
  }
  return x;
}

int main (int argc, char const *argv[])
{
    a_function_to_call();  // Set a breakpoint here to get started 
    second_function(1);
    return 0;
}
