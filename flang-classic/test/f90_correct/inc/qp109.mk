#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test inttoq  ########


qp109: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp109.f08 fcheck.$(OBJX)
	-$(RM) qp109.$(EXESUFFIX) qp109.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp109.f08 -o qp109.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp109.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp109.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp109
	qp109.$(EXESUFFIX)

verify: ;

qp109.run: run

