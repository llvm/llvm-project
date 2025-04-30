#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test dtoq  ########


qp53: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp53.f08 fcheck.$(OBJX)
	-$(RM) qp53.$(EXESUFFIX) qp53.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp53.f08 -o qp53.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp53.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp53.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp53
	qp53.$(EXESUFFIX)

verify: ;

qp53.run: run

