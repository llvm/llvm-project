#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtos  ########


qp68: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp68.f08 fcheck.$(OBJX)
	-$(RM) qp68.$(EXESUFFIX) qp68.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp68.f08 -o qp68.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp68.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp68.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp68
	qp68.$(EXESUFFIX)

verify: ;

qp68.run: run

