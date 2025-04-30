#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test fold_const  ########


qp145: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp145.f08 fcheck.$(OBJX)
	-$(RM) qp145.$(EXESUFFIX) qp145.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp145.f08 -o qp145.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp145.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp145.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp145
	qp145.$(EXESUFFIX)

verify: ;

qp145.run: run

