#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


data01: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/data01.f08 fcheck.$(OBJX)
	-$(RM) data01.$(EXESUFFIX) data01.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/data01.f08 -o data01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) data01.$(OBJX) fcheck.$(OBJX) $(LIBS) -o data01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test data01
	data01.$(EXESUFFIX)

verify: ;

data01.run: run

