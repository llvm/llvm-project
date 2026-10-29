static int globalInt = 123;

// Annotated by two readers at once: ExportAsCore.apinotes, and ExportAs.apinotes
// because ExportAsCore is export_as ExportAs. Each reader is its own slice
// group. See slice-groups.c.
static int sliceGroupProbe = 0;

// Reached by four lookups: a broad one and an exact `Where: Parameters` one
// against each of the two readers. This is the only fixture that combines two
// readers with a parameter selector, so it is the only one where the group
// numbering's ordering property is observable. See slice-groups-order.c.
void sliceGroupOrderProbe(int x);
