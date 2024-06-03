#include <stdlib.h>
#include <stdio.h>
#include <string.h>

char buf[1024];

int fetchTaintedString(char *txt){
  scanf("%s", txt);
}

int exec(char* cmd){
  system(cmd);//warn here
}

void topLevel(){
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  fetchTaintedString (filename);
  strcat(cmd, filename);
  exec(cmd);
}

void printNum(int data){
  printf("Data:%d\n",data);
}