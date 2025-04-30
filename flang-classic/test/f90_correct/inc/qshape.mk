#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test shape function take quadruple precision  ########


qshape: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qshape.f08 fcheck.$(OBJX)
	-$(RM) qshape.$(EXESUFFIX) qshape.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qshape.f08 -o qshape.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qshape.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qshape.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qshape
	qshape.$(EXESUFFIX)

verify: ;

qshape.run: run

