static int static_value = 0;

int
a_function_to_call()
{
    static_value++; // Stop inside the function here.
    return static_value;
}

int main (int argc, char const *argv[])
{
    a_function_to_call();  // Set a breakpoint here to get started 
    return 0;
}
