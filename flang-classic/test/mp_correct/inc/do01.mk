#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do01: do01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do01.$(OBJX): $(SRC)/do01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do01
run: ;
