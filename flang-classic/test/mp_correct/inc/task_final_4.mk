#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
build: task_final_4.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./task_final_4.$(EXESUFFIX) $(LOG)

verify: ;

task_final_4.$(OBJX): $(SRC)/task_final_4.f90 check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/task_final_4.f90
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) task_final_4.$(OBJX) check.$(OBJX) $(LIBS) -o task_final_4.$(EXESUFFIX)

