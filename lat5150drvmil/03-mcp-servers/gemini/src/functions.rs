//! Function calling support

use anyhow::Result;
use serde_json;
use std::fs;
use std::path::Path;

use crate::client::FunctionDeclaration;

pub fn load_functions<P: AsRef<Path>>(path: P) -> Result<Vec<FunctionDeclaration>> {
    let contents = fs::read_to_string(path)?;
    let functions: Vec<FunctionDeclaration> = serde_json::from_str(&contents)?;
    Ok(functions)
}
