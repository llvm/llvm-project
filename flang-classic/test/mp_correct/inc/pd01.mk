#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
pd01: pd01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
pd01.$(OBJX): $(SRC)/pd01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/pd01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) pd01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: pd01
run: ;
