#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par02: par02.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par02.$(OBJX): $(SRC)/par02.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par02.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par02.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par02
run: ;
