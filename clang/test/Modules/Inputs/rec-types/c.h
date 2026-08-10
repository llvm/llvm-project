struct some_descriptor
{
  // commenting line above make this struct work
  void *(*thunk)(struct some_descriptor *);
  unsigned long key;
};

