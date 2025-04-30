#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test quad########


qp46: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp46.f08 fcheck.$(OBJX)
	-$(RM) qp46.$(EXESUFFIX) qp46.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp46.f08 -o qp46.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp46.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp46.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp46
	qp46.$(EXESUFFIX)

verify: ;

qp46.run: run

