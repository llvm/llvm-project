int bottom () { 
  return 1;  // Set a breakpoint here
} 
int foo(int in) { 
  if (in > 0)
    return foo(--in) + 5; 
  else
    return bottom();
}
int main()
{
   return foo(500);
}
