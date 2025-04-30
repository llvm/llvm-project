#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


do01: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/do01.f08 fcheck.$(OBJX)
	-$(RM) do01.$(EXESUFFIX) do01.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/do01.f08 -o do01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) do01.$(OBJX) fcheck.$(OBJX) $(LIBS) -o do01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test do01
	do01.$(EXESUFFIX)

verify: ;

do01.run: run

