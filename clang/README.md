Clang
=====
This is an experimental approach to a frontend for tapir. It allows non-nested
concurrency, allowing interleaved tasks. To do so, it addes the `spawn` and
`sync` keywords, only enabled when `-ftapir` is specified and `<kitsune.h>` is
included. These are used with identifiers to link spawns with sincs:
  
    spawn statement: "spawn" identifier statement
    sync statement: "sync" identifier

For example:
    
    #include<kitsune.h>
    #include<stdio.h> 

    int main(){
      for(int i=0; i<10; i++) spawn pl {
        printf("Hello %d\n", i);
      }
      printf("Done with the loop");
      sync pl; 
    }
    
    
    
