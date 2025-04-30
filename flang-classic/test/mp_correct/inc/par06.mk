#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par06: par06.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par06.$(OBJX): $(SRC)/par06.f  check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par06.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par06.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par06
run: ;
