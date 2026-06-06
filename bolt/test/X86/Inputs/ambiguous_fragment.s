#--- file1
.file "file1.cpp"
.section .text.cold
.type __func.cold.0, @function
__func.cold.0:
  ud2
  .size __func.cold.0, .-__func.cold.0
.section .text
.type __func, @function
__func:
  ud2
  .size __func, .-__func

#--- file2
.file "file2.cpp"
.section .text.cold
.type __func.cold.0, @function
__func.cold.0:
  ud2
  .size __func.cold.0, .-__func.cold.0
.section .text
.type __func, @function
__func:
  ud2
  .size __func, .-__func

#--- file3
.file "file3.cpp"
.section .text.cold
.type __func.cold.0, @function
__func.cold.0:
  ud2
  .size __func.cold.0, .-__func.cold.0
.section .text
.type __func, @function
__func:
  ud2
  .size __func, .-__func

#--- file4
.file "file4.cpp"
.section .text.cold
.type __func.cold.0, @function
__func.cold.0:
  ud2
  .size __func.cold.0, .-__func.cold.0
.section .text
.type __func, @function
__func:
  ud2
  .size __func, .-__func

#--- file5
.file "bolt-pseudo.o"
