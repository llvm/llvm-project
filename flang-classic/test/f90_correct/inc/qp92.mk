#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test sinttoq ########


qp92: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp92.f08 fcheck.$(OBJX)
	-$(RM) qp92.$(EXESUFFIX) qp92.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp92.f08 -o qp92.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp92.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp92.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp92
	qp92.$(EXESUFFIX)

verify: ;

qp92.run: run

