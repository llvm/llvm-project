#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtolog ########


qp87: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp87.f08 fcheck.$(OBJX)
	-$(RM) qp87.$(EXESUFFIX) qp87.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp87.f08 -o qp87.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp87.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp87.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp87
	qp87.$(EXESUFFIX)

verify: ;

qp87.run: run

