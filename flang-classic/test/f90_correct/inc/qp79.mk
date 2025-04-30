#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test logtoq  ########


qp79: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp79.f08 fcheck.$(OBJX)
	-$(RM) qp79.$(EXESUFFIX) qp79.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp79.f08 -o qp79.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp79.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp79.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp79
	qp79.$(EXESUFFIX)

verify: ;

qp79.run: run

